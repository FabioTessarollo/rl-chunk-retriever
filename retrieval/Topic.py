import torch

class Topic:

    def __init__(self, query_emb, page_chunks_dict, ranked_chunks, relevant_chunks, max_exp_loops):
        self.current_rank_chunk = 0 # rank position of the current chunk - to navigate the rank
        self.current_chunk_id = 0 # chunk id value of the current chunk - to navigate the page
        self.query_emb = query_emb
        self.page_chunks_dict = page_chunks_dict
        self.ranked_chunks = ranked_chunks
        self.relevant_chunks = relevant_chunks
        self.bag_of_chunks = []
        self.done = False
        self.max_chunk_id = max(page_chunks_dict.keys())
        self.f1_score = 0
        self.device = torch.device("mps")
        self.bag_of_chunks_embedding = torch.zeros(len(query_emb), dtype=torch.float32, device = self.device)
        self.current_loop = 0
        self.max_exp_loops = max_exp_loops
        self.curr_chunk_emb = None
        self.next_chunk_emb = None
        self.prev_chunk_emb = None

    def cos_sim_norm(self, v1, v2):
        sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        sim = (sim + 1) / 2
        sim = (sim - 0.75) / 0.25
        return sim
    
    def euc_sim_norm(self, v1, v2):
        distance = torch.nn.functional.pairwise_distance(v1.unsqueeze(0), v2.unsqueeze(0))
        sim = torch.exp(-distance / 1.0)
        return sim
    
    def get_state_metadata(self):
        rank_position = (self.current_rank_chunk + 1)  / len(self.ranked_chunks)
        #remaining_loops = (self.current_loop + 1) / self.max_exp_loops
        prev_chunk_already_in_bag = int(self.current_chunk_id - 1 in self.bag_of_chunks)
        single_chunk_already_in_bag = int(self.current_chunk_id in self.bag_of_chunks)
        next_chunk_already_in_bag = int(self.current_chunk_id + 1 in self.bag_of_chunks)
        bag_size = len(self.bag_of_chunks) / len(self.ranked_chunks)
        sq_sim = self.cos_sim_norm(self.curr_chunk_emb, self.query_emb)
        dq_sim = self.cos_sim_norm(self.next_chunk_emb, self.curr_chunk_emb)
        bq_sim = self.cos_sim_norm(self.bag_of_chunks_embedding, self.query_emb)
        pdq_sim = self.cos_sim_norm(self.prev_chunk_emb, self.curr_chunk_emb)
        state_metadata = torch.tensor([rank_position, prev_chunk_already_in_bag, single_chunk_already_in_bag, next_chunk_already_in_bag, bag_size, sq_sim, dq_sim, bq_sim, pdq_sim], device = self.device)
        return state_metadata

    def get_initial_step(self):
        self.current_chunk_id = self.ranked_chunks[0]
        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()
        return (state_embedding, state_metadata, 0, self.done)
    
    def set_f1_score(self):

        bag_size = len(self.bag_of_chunks)
        relevant_size = len(self.relevant_chunks)
        TP = len([c for c in self.bag_of_chunks if c in self.relevant_chunks])
        
        recall = TP / relevant_size if relevant_size > 0 else 0
        precision = TP / bag_size if bag_size > 0 else 0

        self.f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def get_state_embedding(self):

        # get current chunk id embedding
        self.curr_chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)

        # case when current chunk is the last in the page -> no double chunk
        if self.current_chunk_id == self.max_chunk_id:
            self.next_chunk_emb = torch.zeros(768, device = self.device)
        elif self.current_chunk_id == 0:
            self.prev_chunk_emb = torch.zeros(768, device = self.device)

        # get forward double chunk
        if self.current_chunk_id != self.max_chunk_id:
            self.next_chunk_emb = self.page_chunks_dict.get(self.current_chunk_id + 1)

        # get backward double chunk
        if self.current_chunk_id != 0:
            self.prev_chunk_emb = self.page_chunks_dict.get(self.current_chunk_id - 1)
        
        state_embedding = torch.concat((self.curr_chunk_emb, self.next_chunk_emb, self.prev_chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        return state_embedding

    def skip(self):

        if self.current_chunk_id in self.relevant_chunks and self.current_chunk_id not in self.bag_of_chunks:
            reward = -1
        else:
            reward = 1

        if self.current_rank_chunk < 3 and self.current_chunk_id not in self.bag_of_chunks:
            reward -= 1 - self.current_rank_chunk/3

        # restart if last in the rank, else go next
        if self.current_rank_chunk == len(self.ranked_chunks) - 1:
            self.current_loop += 1
            self.current_rank_chunk = 0
            self.current_chunk_id = self.ranked_chunks[0]
        else:
            self.current_rank_chunk += 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]

        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)
    
    def take_single(self):

        # update bag mean embedding
        if self.current_chunk_id not in self.bag_of_chunks:
            n = len(self.bag_of_chunks)
            self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
            self.bag_of_chunks.append(self.current_chunk_id)

        # compute reward
        if self.current_chunk_id in self.relevant_chunks and self.current_chunk_id not in self.bag_of_chunks:
            reward = 2
        elif self.current_chunk_id in self.relevant_chunks:
            reward = 0
        else:
            reward = -1
        
        if self.current_rank_chunk < 3 and self.current_chunk_id not in self.bag_of_chunks:
            reward += 1 - self.current_rank_chunk/3

        # add to bag
        if self.current_chunk_id not in self.bag_of_chunks:
            self.bag_of_chunks.append(self.current_chunk_id)
        
        # go next
        if self.current_rank_chunk == len(self.ranked_chunks) - 1:
            self.current_loop += 1
            self.current_rank_chunk = 0
            self.current_chunk_id = self.ranked_chunks[0]
        else:
            self.current_rank_chunk += 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]

        # new state
        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward/10, self.done)
    
    def take_double(self):

        if self.current_chunk_id == self.max_chunk_id:
            state_embedding, state_metadata, reward, self.done = self.take_single()
            return state_embedding, state_metadata, reward, self.done
        else:

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id + 1

            # update bag mean embedding, if at least one is not already in the bag
            if c1 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.next_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c2)

            # compute reward
            both_relevant = c1 in self.relevant_chunks and c2 in self.relevant_chunks
            #one_is_relevant = c1 in self.relevant_chunks or c2 in self.relevant_chunks
            both_not_in_bag = c1 not in self.bag_of_chunks and c2 not in self.bag_of_chunks
            one_not_in_bag = c1 not in self.bag_of_chunks or c2 not in self.bag_of_chunks
            #one_wasnt_selected_by_cosine_sim = c1 not in self.ranked_chunks[:10] or c2 not in self.ranked_chunks[:10]
            if both_relevant and both_not_in_bag:
                reward = 4 #4
            elif both_relevant and one_not_in_bag:
                reward = 2 #4 forse? provare anche con 2
            elif both_relevant:
                reward = 0
            #elif one_is_relevant and one_not_in_bag:
            #    reward = 0.5
            #elif one_is_relevant:
            #    reward = 0
            else:
                reward = -1
        
        # go next: restart if last -1 in the rank, else go next
        if self.current_rank_chunk == len(self.ranked_chunks) - 1:
            self.current_loop += 1
            self.current_rank_chunk = 0
            self.current_chunk_id = self.ranked_chunks[0]
        else:
            self.current_rank_chunk += 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]

        # new state
        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward/10, self.done)
    
    def take_prev_double(self):

        if self.current_chunk_id == 0:
            state_embedding, state_metadata, reward, self.done = self.take_single()
            return state_embedding, state_metadata, reward, self.done
        else:

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id - 1

            # update bag mean embedding, if at least one is not already in the bag
            if c1 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.prev_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c2)

            # compute reward
            both_relevant = c1 in self.relevant_chunks and c2 in self.relevant_chunks
            one_is_relevant = c1 in self.relevant_chunks or c2 in self.relevant_chunks
            both_not_in_bag = c1 not in self.bag_of_chunks and c2 not in self.bag_of_chunks
            one_not_in_bag = c1 not in self.bag_of_chunks or c2 not in self.bag_of_chunks
            one_wasnt_selected_by_cosine_sim = c1 not in self.ranked_chunks[:10] or c2 not in self.ranked_chunks[:10]
            if both_relevant and both_not_in_bag:
                reward = 4 #4
            elif both_relevant and one_not_in_bag:
                reward = 2 #4 forse? provare anche con 2
            elif both_relevant:
                reward = 0
            #elif one_is_relevant and one_not_in_bag:
            #    reward = 0.5
            #elif one_is_relevant:
            #    reward = 0
            else:
                reward = -1
        
        # go next: restart if last -1 in the rank, else go next
        if self.current_rank_chunk == len(self.ranked_chunks) - 1:
            self.current_loop += 1
            self.current_rank_chunk = 0
            self.current_chunk_id = self.ranked_chunks[0]
        else:
            self.current_rank_chunk += 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]

        # new state
        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward/10, self.done)

    def submit_current_bag(self):

        self.done = True

        self.set_f1_score()

        if len(self.bag_of_chunks) == 0:
            reward = -0.1
        else:
            reward = self.f1_score

        self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)