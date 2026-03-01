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
        self.truncated = False
        self.max_chunk_id = max(page_chunks_dict.keys())
        self.f1_score = 0
        self.recall = 0
        self.precision = 0
        self.device = torch.device("mps")
        self.bag_of_chunks_embedding = torch.zeros(len(query_emb), dtype=torch.float32, device = self.device)
        self.current_loop = 0
        self.max_exp_loops = max_exp_loops
        self.curr_chunk_emb = None
        self.next_chunk_emb = None
        self.prev_chunk_emb = None
        
    def _advance_rank(self):
        if self.current_rank_chunk == len(self.ranked_chunks) - 1:
            self.set_f1_score()
            self.truncated = True
        else:
            self.current_rank_chunk += 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
            while self.current_chunk_id in self.bag_of_chunks:
                if self.current_rank_chunk == len(self.ranked_chunks) - 1:
                    self.set_f1_score()
                    self.truncated = True
                    break
                self.current_rank_chunk += 1
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]


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
        #prev_chunk_already_in_bag = int(self.current_chunk_id - 1 in self.bag_of_chunks)
        #single_chunk_already_in_bag = int(self.current_chunk_id in self.bag_of_chunks)
        #next_chunk_already_in_bag = int(self.current_chunk_id + 1 in self.bag_of_chunks)
        bag_size = len(self.bag_of_chunks) / len(self.ranked_chunks)
        sq_sim = self.cos_sim_norm(self.curr_chunk_emb, self.query_emb)
        dq_sim = self.cos_sim_norm(self.next_chunk_emb, self.curr_chunk_emb)
        bq_sim = self.cos_sim_norm(self.bag_of_chunks_embedding, self.query_emb)
        pdq_sim = self.cos_sim_norm(self.prev_chunk_emb, self.curr_chunk_emb)
        state_metadata = torch.tensor([rank_position, bag_size, sq_sim, dq_sim, bq_sim, pdq_sim], device = self.device)
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
        
        self.recall = TP / relevant_size if relevant_size > 0 else 0
        self.precision = TP / bag_size if bag_size > 0 else 0

        self.f1_score = (2 * self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0
    
    def get_state_embedding(self):

        # get current chunk id embedding
        self.curr_chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)

        if self.current_chunk_id == self.max_chunk_id:
            self.next_chunk_emb = torch.zeros(768, device = self.device)
        else:
            self.next_chunk_emb = self.page_chunks_dict.get(self.current_chunk_id + 1)

        if self.current_chunk_id == 0:
            self.prev_chunk_emb = torch.zeros(768, device = self.device)
        else:
            self.prev_chunk_emb = self.page_chunks_dict.get(self.current_chunk_id - 1)
        
        state_embedding = torch.concat((self.curr_chunk_emb, self.next_chunk_emb, self.prev_chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        return state_embedding

    def skip(self):

        if self.current_chunk_id in self.relevant_chunks:
            reward = -1
            if self.current_rank_chunk < 3 and self.current_chunk_id not in self.bag_of_chunks:
                reward -= (1 - self.current_rank_chunk/3)
        else:
            reward = 0

        self._advance_rank()

        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward/10, self.done, self.truncated)
    
    def take_single(self):

        # compute reward
        if self.current_chunk_id in self.relevant_chunks:
            reward = 1
        else:
            reward = -1

        # update bag mean embedding
        if self.current_chunk_id not in self.bag_of_chunks:
            n = len(self.bag_of_chunks)
            self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
            self.bag_of_chunks.append(self.current_chunk_id)
        
        self._advance_rank()

        # new state
        state_embedding = self.get_state_embedding()
        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward/10, self.done, self.truncated)
    
    def take_double(self):

        if self.current_chunk_id == self.max_chunk_id:
            state_embedding, state_metadata, reward, self.done, self.trunacted = self.take_single()
            return state_embedding, state_metadata, reward, self.done, self.trunacted
        else:

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id + 1

            # compute reward
            both_relevant = c1 in self.relevant_chunks and c2 in self.relevant_chunks
            one_is_relevant = c1 in self.relevant_chunks or c2 in self.relevant_chunks
            if both_relevant:
                reward = 2
            elif one_is_relevant:
                reward = 0
            else:
                reward = -2

            # update bag mean embedding, if at least one is not already in the bag
            if c1 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.next_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c2)
        
            self._advance_rank()

            # new state
            state_embedding = self.get_state_embedding()
            state_metadata = self.get_state_metadata()

            return (state_embedding, state_metadata, reward/10, self.done, self.truncated)
    
    def take_prev_double(self):

        if self.current_chunk_id == 0:
            state_embedding, state_metadata, reward, self.done, self.trunacted = self.take_single()
            return state_embedding, state_metadata, reward, self.done, self.trunacted
        else:

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id - 1

            # compute reward
            both_relevant = c1 in self.relevant_chunks and c2 in self.relevant_chunks
            one_is_relevant = c1 in self.relevant_chunks or c2 in self.relevant_chunks
            if both_relevant:
                reward = 2
            elif one_is_relevant:
                reward = 0
            else:
                reward = -2

            # update bag mean embedding, if at least one is not already in the bag
            if c1 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.prev_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c2)
        
            self._advance_rank()

            # new state
            state_embedding = self.get_state_embedding()
            state_metadata = self.get_state_metadata()

            return (state_embedding, state_metadata, reward/10, self.done, self.truncated)
        
    def take_triple(self):

        if self.current_chunk_id == 0 or self.current_chunk_id == self.max_chunk_id:
            state_embedding, state_metadata, reward, self.done, self.trunacted = self.take_single()
            return state_embedding, state_metadata, reward, self.done, self.trunacted
        else:

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id - 1
            c3 = self.current_chunk_id + 1

            # compute reward
            selected = [c1, c2, c3]
            relevant_selected = sum(1 for c in selected if c in self.relevant_chunks)

            if relevant_selected == 3:
                reward = 3
            elif relevant_selected == 2:
                reward = 1
            elif relevant_selected == 1:
                reward = -1
            else:  
                reward = -3

            # update bag mean embedding, if at least one is not already in the bag
            if c1 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.curr_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.prev_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c2)
            if c3 not in self.bag_of_chunks:
                n = len(self.bag_of_chunks)
                self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * n + self.next_chunk_emb) / (n + 1)
                self.bag_of_chunks.append(c3)
        
            self._advance_rank()

            # new state
            state_embedding = self.get_state_embedding()
            state_metadata = self.get_state_metadata()

            return (state_embedding, state_metadata, reward/10, self.done, self.truncated)