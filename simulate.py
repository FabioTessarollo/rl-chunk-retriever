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


def pprint(topic, reward, state_metadata, action):

    print(f"Last Action: {action}")
    print(f"current_chunk_id: {topic.current_chunk_id}")
    print(f"current_rank_chunk: {topic.current_rank_chunk}")
    sm = state_metadata.tolist()
    labels = [
        "rank_position",
        "remaining_loops",
        "single_chunk_already_in_bag",
        "next_chunk_already_in_bag",
        "bag_size",
        "sq_sim",
        "dq_sim",
        "bq_sim"
    ]

    print("State metadata:")
    for label, value in zip(labels, sm):
        print(f"  {label:35}: {value}")
    print(f"Reward: {reward}")


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

    query_id = training_set[30] #30
    query = data.get_query_obj_from_id(query_id)
    page_id = query.get("page_id")
    page, page_even, page_odd = data.get_page_chunks_dict(page_id)
    query_emb = torch.tensor(query.get("query")).to(device)
    query_desc = query.get("query_desc")
    relevant_chunks = query.get("relevant_chunks")
    ranked_chunks = data.get_ranked_with_prev_chunks_from_query_id(query_id)
    #ranked_chunks = data.cosine_sim_rank[str(query_id)]


    print(f"Query Desc: {query_desc}")
    print(f"Relevant Chunks: {relevant_chunks}")
    #print(f"Original Ranked Chunks: {sorted(orig_ranked_chunks)}")
    print(f"Ranked Chunks.        : {ranked_chunks}")

    topic = Topic(query_emb, page, page_even, page_odd, ranked_chunks, relevant_chunks, max_exp_loops)
    
    _, state_meta, reward, _ = topic.get_initial_step()
    pprint(topic, reward, state_meta, 'init')
    _, state_meta, reward, done = topic.take_single()
    pprint(topic, reward, state_meta, 'take_single')
    _, state_meta, reward, done = topic.take_single()
    pprint(topic, reward, state_meta, 'take_single')
    _, state_meta, reward, done = topic.skip()
    pprint(topic, reward, state_meta, 'skip')
    _, state_meta, reward, done = topic.take_double()
    pprint(topic, reward, state_meta, 'take_double')
    _, state_meta, reward, done = topic.submit_current_bag()
    pprint(topic, reward, state_meta, 'submit_current_bag')

    print(f"Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Actions: {topic.actions_taken}")
        
if __name__ == "__main__":
    main()