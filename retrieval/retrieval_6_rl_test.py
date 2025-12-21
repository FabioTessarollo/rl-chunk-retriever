import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

from retrieval.Data import Data
from retrieval.Topic import Topic
from retrieval.DuelingDQN import DuelingDQN
from retrieval.ReplayBuffer import PrioritizedReplayBuffer
from retrieval.EarlyStopping import EarlyStopping


now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
random.seed(1)
torch.manual_seed(1)

def evaluate(data, query_ids, online_net, device, max_exp_loops):
    """Evaluate the model on validation set without training"""
    val_reward = 0
    val_f1_score = 0
    results = []
    
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        page = data.get_page_chunks_dict(page_id)
        query_emb = torch.tensor(query.get("query")).to(device)
        relevant_chunks = query.get("relevant_chunks")
        ranked_chunks = data.cosine_sim_rank[str(query_id)]
        query_desc = query.get("query_desc")

        topic = Topic(query_emb, page, ranked_chunks, relevant_chunks, max_exp_loops)

        state_emb, state_meta, _, _ = topic.get_initial_step()
        state_emb = state_emb.to(device)
        state_meta = state_meta.to(device)
        episode_reward = 0
        done = False
        episode_steps = 0
        
        # Greedy evaluation (no exploration)
        while not done:
            episode_steps += 1
            
            with torch.no_grad():
                q = online_net(state_emb.unsqueeze(0), state_meta.unsqueeze(0))
                action = q.argmax().item()
            
                if topic.current_loop + 1 > max_exp_loops:
                    next_emb, next_meta, reward, done = topic.submit_current_bag()
                elif action == 0:
                    next_emb, next_meta, reward, done = topic.skip()
                elif action == 1:
                    next_emb, next_meta, reward, done = topic.take_single()
                elif action == 2:
                    next_emb, next_meta, reward, done = topic.take_double()
                elif action == 3:
                    next_emb, next_meta, reward, done = topic.take_prev_double()
                elif action == 4:
                    next_emb, next_meta, reward, done = topic.submit_current_bag()
                
            next_emb = next_emb.to(device)
            next_meta = next_meta.to(device)
            episode_reward += reward
            
            state_emb, state_meta = next_emb, next_meta

            logging.info(f"Step: {episode_steps}, Reward: {reward:.4f}, Action: {action}")
            
        val_reward += episode_reward
        val_f1_score += topic.f1_score

        results.append({
            "query" : query_desc,
            "rl_f1_score" : topic.f1_score,
            "rl_model_retrieved" : sorted(topic.bag_of_chunks)

        })

        logging.info(f"GREEDY - Query: {query_desc}, Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}, Actions: {topic.actions_taken}")
    
    avg_val_reward = val_reward / len(query_ids)
    avg_val_f1_score = val_f1_score / len(query_ids)
    
    return avg_val_reward, avg_val_f1_score, results

def main():

    pages_path_test = f"data_chunks_emb/pages_chunked_emb_test.json"
    relevant_path_test = f"data_chunks_emb/relevant_chunks_emb_test.json"
    cosine_sim_path_test = "data_chunks_cos_sim/cosine_sim_rank_threshold_only_single_test.json"

    data_test = Data(pages_path_test, relevant_path_test, cosine_sim_path_test)
    data_test.load_pages()
    data_test.load_relevant()
    data_test.load_cosine_sim()

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DuelingDQN(metadata_dim = 9, action_dim = 5, proj_dim = 512, dropout_p = 0).to(device)
    model.load_state_dict(torch.load("models/rl-chunk-retriever.pt", map_location="cpu")) #rl-chunk-retriever copy
    model.eval()

    avg_val_reward, avg_val_f1_score, results = evaluate(
        data_test, data_test.query_ids, model, device, 
        max_exp_loops = 1
    )
    logging.info(f"GREEDY: TEST Reward: {avg_val_reward:.4f}, TEST F1: {avg_val_f1_score:.4f}")
    print(f"GREEDY: TEST - Reward: {avg_val_reward:.4f}, F1: {avg_val_f1_score:.4f}")

    output_dir = "data_analysis"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "rl_model_retrieved_test_single.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()