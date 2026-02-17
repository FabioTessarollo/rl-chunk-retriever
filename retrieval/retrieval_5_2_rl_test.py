import json
import os
import random
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Must be called before importing plt
import matplotlib.pyplot as plt
import numpy as np

from retrieval.Data import Data
from retrieval.Topic import Topic
from retrieval.DuelingDQN import DuelingDQN


now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
random.seed(1)
torch.manual_seed(1)


def compute_stream_feature_importance(online_net, data, query_ids, device, max_exp_loops, n_samples=1000):
    import os
    online_net.eval()

    metadata_feature_names = [
        "Rank Position",
        "Prev Chunk Taken",
        "Curr Chunk Taken",
        "Next Chunk Taken",
        "Bag of Chunks Size",
        "Curr and Query Sim",
        "Next and Query Sim",
        "Bag and Query Sim",
        "Prev and Query Sim"
    ]
    emb_group_names = ["Emb - Current", "Emb - Current & Next", "Emb - Current & Prev", "Emb - Query", "Emb - Bag"]

    sample_n = min(len(query_ids), n_samples)
    sampled_qids = random.sample(list(query_ids), sample_n)

    emb_sum_v = None
    emb_sum_a = None
    meta_sum_v = None
    meta_sum_a = None
    total_states = 0

    for qid in sampled_qids:
        query = data.get_query_obj_from_id(qid)
        page_id = query.get("page_id")
        page = data.get_page_chunks_dict(page_id)
        query_emb = torch.tensor(query.get("query")).to(device)
        relevant_chunks = query.get("relevant_chunks")
        ranked_chunks = data.cosine_sim_rank[str(qid)]
        topic = Topic(query_emb, page, ranked_chunks, relevant_chunks, max_exp_loops)

        state_emb, state_meta, _, _ = topic.get_initial_step()
        state_emb = state_emb.to(device)
        state_meta = state_meta.to(device)
        done = False

        while not done:
            state_emb_var = state_emb.clone().detach().unsqueeze(0).requires_grad_(True)
            state_meta_var = state_meta.clone().detach().unsqueeze(0).requires_grad_(True)

            q_vals, v_stream, a_stream = online_net(
                state_emb_var, state_meta_var, return_streams=True
            )

            v_scalar = v_stream.sum()
            a_scalar = a_stream.abs().sum()

            online_net.zero_grad()
            v_scalar.backward(retain_graph=True)
            emb_grad_v = state_emb_var.grad.detach().squeeze(0).abs().cpu()
            meta_grad_v = state_meta_var.grad.detach().squeeze(0).abs().cpu()

            state_emb_var.grad.zero_()
            state_meta_var.grad.zero_()

            a_scalar.backward()
            emb_grad_a = state_emb_var.grad.detach().squeeze(0).abs().cpu()
            meta_grad_a = state_meta_var.grad.detach().squeeze(0).abs().cpu()

            if emb_sum_v is None:
                emb_sum_v = emb_grad_v.clone()
                emb_sum_a = emb_grad_a.clone()
                meta_sum_v = meta_grad_v.clone()
                meta_sum_a = meta_grad_a.clone()
            else:
                emb_sum_v += emb_grad_v
                emb_sum_a += emb_grad_a
                meta_sum_v += meta_grad_v
                meta_sum_a += meta_grad_a

            total_states += 1

            action = q_vals.argmax().item()
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
            else:
                next_emb, next_meta, reward, done = topic.submit_current_bag()

            state_emb = next_emb.to(device)
            state_meta = next_meta.to(device)

    if total_states == 0:
        return

    emb_mean_v = emb_sum_v / total_states
    emb_mean_a = emb_sum_a / total_states
    meta_mean_v = (meta_sum_v / total_states).numpy()
    meta_mean_a = (meta_sum_a / total_states).numpy()

    emb_len = emb_mean_v.numel()
    groups = len(emb_group_names)
    group_size = emb_len // groups

    emb_mean_v_groups = emb_mean_v.view(groups, group_size).mean(dim=1).numpy()
    emb_mean_a_groups = emb_mean_a.view(groups, group_size).mean(dim=1).numpy()

    labels = metadata_feature_names + emb_group_names
    value_scores = np.concatenate([meta_mean_v, emb_mean_v_groups])
    adv_scores = np.concatenate([meta_mean_a, emb_mean_a_groups])

    sort_idx = np.argsort(adv_scores)
    labels = [labels[i] for i in sort_idx]
    value_scores = value_scores[sort_idx]
    adv_scores = adv_scores[sort_idx]

    y = np.arange(len(labels))
    height = 0.4

    os.makedirs("feature_importance", exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.barh(y - height/2, value_scores, height=height, label="Value")
    plt.barh(y + height/2, adv_scores, height=height, label="Advantage")
    plt.yticks(y, labels)
    plt.xlabel("Average Gradient")
    plt.title("Value vs Advantage Stream Feature Importance")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("value_vs_advantage_importance.png")
    plt.close()

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

        logging.info(f"Query: {query_desc}")

        state_emb, state_meta, _, _ = topic.get_initial_step()
        state_emb = state_emb.to(device)
        state_meta = state_meta.to(device)
        episode_reward = 0
        done = False
        episode_steps = 0
        
        # Greedy evaluation (no exploration)
        while not done:
            episode_steps += 1

            log_state = f"Step: {episode_steps}, Chunk: {topic.current_chunk_id}"
            
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

            logging.info(f"{log_state}, Reward: {reward:.4f}, Action: {action}")
            
        val_reward += episode_reward
        val_f1_score += topic.f1_score

        results.append({
            "query" : query_desc,
            "rl_f1_score" : topic.f1_score,
            "rl_model_retrieved" : sorted(topic.bag_of_chunks)

        })

        logging.info(f"Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}")
    
    avg_val_reward = val_reward / len(query_ids)
    avg_val_f1_score = val_f1_score / len(query_ids)
    
    return avg_val_reward, avg_val_f1_score, results

def test():

    logging.basicConfig(filename='rl_testing.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode="w")

    pages_path_test = f"data_3_embed/pages_chunked_emb_test.json"
    relevant_path_test = f"data_3_embed/relevant_chunks_emb_test.json"
    cosine_sim_path_test = "data_4_cos_sim/cosine_sim_rank_threshold_only_single_test.json"

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

    output_dir = "data_5_analysis"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "rl_model_retrieved_test_single.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    compute_stream_feature_importance(
        model, data_test, data_test.query_ids, device, max_exp_loops = 1, n_samples=200 #  + validation_set
    )

