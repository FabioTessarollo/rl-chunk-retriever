import torch

"""
 state, and input of the deep Q-learning model:
 - embedding: current_chunk 
 - embedding: current_bag_of_selected_chunks (computed as the mean embedding of the selected chunks)
 - embedding: query
 - list: metadata

actions, and output of the deep Q-learning model (Q-values), to let it explore the cosine similarity rank:
 - get_next_chunk_in_rank
 - get_prev_chunk_in_rank
 - get_fw_extended_chunk (current chunk and next in the page, embedded together)
 - get_bw_extended_chunk (current chunk and previous in the page, embedding together)
 - add_chunk_to_bag
 - submit_current_bag
 """

class Topic:

    def __init__(self, query_emb, page_chunks_dict, page_even_chunks_dict, page_odd_chunks_dict, ranked_chunks, relevant_chunks, max_exp_steps):
        self.current_rank_chunk = 0 # rank position of the current chunk - to navigate the rank
        self.current_chunk_id = 0 # chunk id value of the current chunk - to navigate the page
        self.query_emb = query_emb
        self.page_chunks_dict = page_chunks_dict
        self.page_even_chunks_dict = page_even_chunks_dict #extended embeddings
        self.page_odd_chunks_dict = page_odd_chunks_dict #extended embeddings
        self.ranked_chunks = ranked_chunks
        self.relevant_chunks = relevant_chunks
        self.bag_of_chunks = []
        self.actions_taken = {'nr': 0, 'pr': 0, 'fw': 0, 'bw': 0, 'ad': 0, 'su': 0}
        self.last_action = None
        self.done = False
        self.max_chunk_id = max(page_chunks_dict.keys())
        self.chunk_emb = None
        self.f1_score = 0
        self.device = torch.device("mps")
        self.bag_of_chunks_embedding = torch.zeros(len(query_emb), dtype=torch.float32, device = self.device)
        self.exp_steps = 0
        self.max_exp_steps = max_exp_steps
    
    def get_state_metadata(self): # add multiple scores
        rank_position = self.current_rank_chunk  / len(self.ranked_chunks)
        remaining_steps = (self.max_exp_steps - self.exp_steps) / self.max_exp_steps
        #page_position = self.current_chunk_id / self.max_chunk_id

        fw_action = 0
        bw_action = 0
        ext_chunk_already_in_bag = 0
        if self.last_action == 'fw':
            fw_action = 1
            c1 = self.current_chunk_id
            c2 = self.current_chunk_id + 1
            if c1 in self.bag_of_chunks and c2 in self.bag_of_chunks:
                ext_chunk_already_in_bag = 1
            else:
                ext_chunk_already_in_bag = 0
        elif self.last_action == 'bw':
            bw_action = 1
            c1 = self.current_chunk_id
            c2 = self.current_chunk_id - 1
            if c1 in self.bag_of_chunks and c2 in self.bag_of_chunks:
                ext_chunk_already_in_bag = 1              
        
        single_chunk_already_in_bag = int(self.current_chunk_id in self.bag_of_chunks)
        
        last_action_is_add = int(self.last_action == 'ad')

        bag_size = len(self.bag_of_chunks)/10
        state_metadata = torch.tensor([rank_position, fw_action, bw_action, last_action_is_add, single_chunk_already_in_bag, ext_chunk_already_in_bag, bag_size, remaining_steps], device = self.device)
        return state_metadata
    
    def discouraged_action(self, action_code, reward):
        self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)
        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)
        self.last_action = action_code
        self.actions_taken[action_code] += 1
        state_metadata = self.get_state_metadata()
        return (state_embedding, state_metadata, reward, self.done)

    def get_initial_step(self):
        self.current_chunk_id = self.ranked_chunks[0]
        self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)
        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)
        state_metadata = self.get_state_metadata()
        return (state_embedding, state_metadata, 0, self.done)

    def get_next_chunk_in_rank(self):
        self.exp_steps+=1
        
        self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
        
        # if the skipped chunk was relevant and not already taken, penalty
        if self.current_chunk_id in self.relevant_chunks and self.current_chunk_id not in self.bag_of_chunks:
            reward = -0.1
        else:
            reward = 0

        if self.current_rank_chunk == len(self.ranked_chunks) - 1:
            self.current_chunk_id = self.ranked_chunks[0]
        else:
            self.current_rank_chunk += 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
        
        self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)

        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        self.last_action = 'nr'
        self.actions_taken['nr'] += 1

        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)
    
    def get_prev_chunk_in_rank(self):
        self.exp_steps+=1

        self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]

        if self.current_rank_chunk == 0:
            return self.discouraged_action('pr', 0)
        
        # if the skipped chunk was relevant and not already taken, penalty
        if self.current_chunk_id in self.relevant_chunks and self.current_chunk_id not in self.bag_of_chunks:
            reward = -0.1
        else:
            reward = 0

        if self.current_rank_chunk == 0:
            self.current_chunk_id = self.ranked_chunks[len(self.ranked_chunks) - 1]
        else:
            self.current_rank_chunk -= 1
            self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]    
        
        self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)

        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        self.last_action = 'pr'
        self.actions_taken['pr'] += 1

        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)

    def get_fw_extended_chunk(self):
        self.exp_steps+=1

        if self.last_action == 'fw' or self.last_action == 'ad':
            reward = -1
        else:
            reward = 0

        if self.current_chunk_id == self.max_chunk_id:
            self.current_chunk_id -= 1

        if self.current_chunk_id % 2 == 0 or self.current_chunk_id == 0:
            self.chunk_emb = self.page_even_chunks_dict.get(self.current_chunk_id)
        else:
            self.chunk_emb = self.page_odd_chunks_dict.get(self.current_chunk_id)

        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        self.last_action = 'fw'
        self.actions_taken['fw'] += 1

        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)

    def get_bw_extended_chunk(self):
        self.exp_steps+=1

        if self.last_action == 'fw' or self.last_action == 'ad':
            reward = -1
        else:
            reward = 0

        if self.current_chunk_id < 1:
            self.current_chunk_id += 1

        if self.current_chunk_id % 2 == 0 or self.current_chunk_id == 0:
            temp_chunk_id = self.current_chunk_id -1
            self.chunk_emb = self.page_odd_chunks_dict.get(temp_chunk_id)
        else:
            temp_chunk_id = self.current_chunk_id -1
            self.chunk_emb = self.page_even_chunks_dict.get(temp_chunk_id)

        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        self.last_action = 'bw'
        self.actions_taken['bw'] += 1

        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)

    def add_chunk_to_bag(self):
        self.exp_steps+=1

        if self.last_action == 'ad':
            return self.discouraged_action('ad', -1)
        

        # update bag mean embedding
        if len(self.bag_of_chunks) > 0:
            self.bag_of_chunks_embedding = (self.bag_of_chunks_embedding * len(self.bag_of_chunks) + self.chunk_emb) / (len(self.bag_of_chunks) + 1)
        else:
            self.bag_of_chunks_embedding = self.chunk_emb

        # add to list of selected chunks
        if self.last_action == 'fw':

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id + 1

            if c1 in self.relevant_chunks and c1 not in self.bag_of_chunks and c2 in self.relevant_chunks and c2 not in self.bag_of_chunks and c2 not in self.ranked_chunks:
                reward = 5
            elif c1 in self.relevant_chunks and c1 not in self.bag_of_chunks and c2 in self.relevant_chunks and c2 not in self.bag_of_chunks:
                reward = 3
            elif c1 in self.relevant_chunks and c2 in self.relevant_chunks:
                reward = 0
            elif c1 in self.relevant_chunks and c2 not in self.relevant_chunks:
                reward = -1
            else:
                reward = -2
            
            if c1 not in self.bag_of_chunks:
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                self.bag_of_chunks.append(c2)

            # move to next
            if self.current_rank_chunk + 2 >= len(self.ranked_chunks) - 1:
                self.current_rank_chunk = 0
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
                self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)
            else:
                self.exp_steps+=1
                self.current_rank_chunk += 2
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
                self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)
        
        elif self.last_action == 'bw':

            c1 = self.current_chunk_id
            c2 = self.current_chunk_id - 1

            if c1 in self.relevant_chunks and c1 not in self.bag_of_chunks and c2 in self.relevant_chunks and c2 not in self.bag_of_chunks and c2 not in self.ranked_chunks:
                reward = 5
            elif c1 in self.relevant_chunks and c1 not in self.bag_of_chunks and c2 in self.relevant_chunks and c2 not in self.bag_of_chunks:
                reward = 3
            elif c1 in self.relevant_chunks and c2 in self.relevant_chunks:
                reward = 0
            elif c1 in self.relevant_chunks and c2 not in self.relevant_chunks:
                reward = -1
            else:
                reward = -2
            
            if c1 not in self.bag_of_chunks:
                self.bag_of_chunks.append(c1)
            if c2 not in self.bag_of_chunks:
                self.bag_of_chunks.append(c2)
            
            # move to next
            if self.current_rank_chunk == len(self.ranked_chunks) - 1:
                self.current_rank_chunk = 0
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
                self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)
            else:
                self.current_rank_chunk += 1
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
                self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)

        # not extended case
        else:
            
            if self.current_chunk_id in self.relevant_chunks and self.current_chunk_id not in self.bag_of_chunks:
                reward = 2
            elif self.current_chunk_id in self.relevant_chunks and self.current_chunk_id in self.bag_of_chunks:
                reward = 0
            else:
                reward = -1

            if self.current_chunk_id not in self.bag_of_chunks:
                self.bag_of_chunks.append(self.current_chunk_id)

            # move to next
            if self.current_rank_chunk == len(self.ranked_chunks) - 1:
                self.current_rank_chunk = 0
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
                self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)
            else:
                self.current_rank_chunk += 1
                self.current_chunk_id = self.ranked_chunks[self.current_rank_chunk]
                self.chunk_emb = self.page_chunks_dict.get(self.current_chunk_id)

        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        self.last_action = 'ad'
        self.actions_taken['ad'] += 1

        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)

    def submit_current_bag(self):

        self.done = True

        if self.last_action is None or self.actions_taken['ad'] == 0 or len(self.bag_of_chunks) == 0:
            return self.discouraged_action('su', -1)

        bag_size = len(self.bag_of_chunks)
        relevant_size = len(self.relevant_chunks)
        TP = len([c for c in self.bag_of_chunks if c in self.relevant_chunks])
        
        recall = TP / relevant_size if relevant_size > 0 else 0
        precision = TP / bag_size if bag_size > 0 else 0

        self.f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        state_embedding = torch.concat((self.chunk_emb, self.query_emb, self.bag_of_chunks_embedding), dim = 0)

        self.last_action = 'su'
        self.actions_taken['su'] += 1

        reward = self.f1_score * 50

        state_metadata = self.get_state_metadata()

        return (state_embedding, state_metadata, reward, self.done)