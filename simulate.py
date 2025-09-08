import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from Data import Data
from Topic import Topic
from DuelingDQN import DuelingDQN
from ReplayBuffer import PrioritizedReplayBuffer

random.seed(1)
torch.manual_seed(1)

def main():
    pages_path = "data_chunks_emb/pages_chunked_emb.json"
    pages_doub_even_path = "data_chunks_emb/pages_doub_chunked_even.json"
    pages_doub_odd_path = "data_chunks_emb/pages_doub_chunked_odd.json"
    relevant_path = "data_chunks_emb/relevant_chunks_emb.json"
    cosine_sim_path = "data_chunks_cos_sim/cosine_sim_rank.json"

    data = Data(pages_path, relevant_path, pages_doub_even_path, pages_doub_odd_path, cosine_sim_path)
    data.load_pages()
    data.load_pages_even()
    data.load_pages_odd()
    data.load_relevant()
    data.load_cosine_sim()

    fair_query_ids, difficult_query_ids = data.get_query_ids_by_difficulty()

    training_set, validation_set = data.split_query_ids(fair_query_ids, 0.7)

    max_exp_loops=30

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    training_set = sorted(training_set)

    query_id = training_set[1]
    query = data.get_query_obj_from_id(query_id)
    page_id = query.get("page_id")
    page, page_even, page_odd = data.get_page_chunks_dict(page_id)
    query_emb = torch.tensor(query.get("query")).to(device)
    query_desc = query.get("query_desc")
    relevant_chunks = query.get("relevant_chunks")
    ranked_chunks = data.cosine_sim_rank[str(query_id)]

    print(f"Query Desc: {query_desc}")
    print(f"Relevant Chunks: {relevant_chunks}")
    print(f"Ranked Chunks: {ranked_chunks}")

    topic = Topic(query_emb, page, page_even, page_odd, ranked_chunks, relevant_chunks, max_exp_loops)
    
    _, state_meta, reward, _ = topic.get_initial_step()
    _, state_meta, reward, done = topic.skip()
    print(topic.current_chunk_id in topic.relevant_chunks and topic.current_chunk_id not in topic.bag_of_chunks)
    _, state_meta, reward, done = topic.take_single()
    print(reward)
    print(topic.current_chunk_id)
    _, state_meta, reward, done = topic.skip()
    _, state_meta, reward, done = topic.skip()
    _, state_meta, reward, done = topic.skip()
    _, state_meta, reward, done = topic.skip()
    _, state_meta, reward, done = topic.take_single()
    print(reward)
    _, state_meta, reward, done = topic.submit_current_bag()
    print(reward)

    print(f"Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Actions: {topic.actions_taken}")
        
if __name__ == "__main__":
    main()