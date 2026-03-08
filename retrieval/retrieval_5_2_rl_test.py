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
random.seed(39)
torch.manual_seed(39)

def compute_stream_ablation_importance(online_net, data, query_ids, device, max_exp_loops, n_samples=1000):
    import os
    online_net.eval()

    metadata_feature_names = [
        "Rank Position",
        "Bag of Chunks Size",
        "Query - Current Chunk Sim", "Query - Current Chunk and Next Sim", "Query - Bag Sim", "Query - Current Chunk and Prev Sim"
    ]
    emb_group_names = ["Emb - Current", "Emb - Current & Next", "Emb - Current & Prev", "Emb - Query", "Emb - Bag"]

    sample_n = min(len(query_ids), n_samples)
    sampled_qids = random.sample(list(query_ids), sample_n)

    # Per-feature accumulators for Q-drop and action-flip
    n_meta = len(metadata_feature_names)
    n_emb_groups = len(emb_group_names)

    meta_q_drop = np.zeros(n_meta)
    meta_flip   = np.zeros(n_meta)
    emb_q_drop  = np.zeros(n_emb_groups)
    emb_flip    = np.zeros(n_emb_groups)
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
        truncated = False

        while not done and not truncated:
            with torch.no_grad():
                # --- Baseline ---
                q_base, _, _ = online_net(
                    state_emb.unsqueeze(0), state_meta.unsqueeze(0), return_streams=True
                )
                q_base_vals  = q_base.squeeze(0)
                action_base  = q_base_vals.argmax().item()
                q_base_max   = q_base_vals[action_base].item()

                # --- Ablate each metadata feature individually ---
                for i in range(n_meta):
                    meta_ablated = state_meta.clone()
                    meta_ablated[i] = 0.0
                    q_abl, _, _ = online_net(
                        state_emb.unsqueeze(0), meta_ablated.unsqueeze(0), return_streams=True
                    )
                    q_abl_vals = q_abl.squeeze(0)
                    action_abl = q_abl_vals.argmax().item()
                    meta_q_drop[i] += q_base_max - q_abl_vals[action_base].item()
                    meta_flip[i]   += int(action_abl != action_base)

                # --- Ablate each embedding group individually ---
                emb_len    = state_emb.numel()
                groups     = n_emb_groups
                assert emb_len % groups == 0, (
                    f"Embedding length {emb_len} not divisible by {groups} groups."
                )
                group_size = emb_len // groups

                for i in range(n_emb_groups):
                    emb_ablated = state_emb.clone()
                    emb_ablated[i * group_size : (i + 1) * group_size] = 0.0
                    q_abl, _, _ = online_net(
                        emb_ablated.unsqueeze(0), state_meta.unsqueeze(0), return_streams=True
                    )
                    q_abl_vals = q_abl.squeeze(0)
                    action_abl = q_abl_vals.argmax().item()
                    emb_q_drop[i] += q_base_max - q_abl_vals[action_base].item()
                    emb_flip[i]   += int(action_abl != action_base)

            total_states += 1

            # --- Step environment using baseline action ---
            if action_base == 0:
                next_emb, next_meta, reward, done, truncated = topic.skip()
            elif action_base == 1:
                next_emb, next_meta, reward, done, truncated = topic.take_single()
            elif action_base == 2:
                next_emb, next_meta, reward, done, truncated = topic.take_double()
            elif action_base == 3:
                next_emb, next_meta, reward, done, truncated = topic.take_prev_double()
            elif action_base == 4:
                next_emb, next_meta, reward, done, truncated = topic.take_triple()

            state_emb  = next_emb.to(device)
            state_meta = next_meta.to(device)

    if total_states == 0:
        return

    # --- Normalize ---
    meta_q_drop_mean = meta_q_drop / total_states
    meta_flip_rate   = meta_flip   / total_states
    emb_q_drop_mean  = emb_q_drop  / total_states
    emb_flip_rate    = emb_flip    / total_states

    labels      = metadata_feature_names + emb_group_names
    q_drop_all  = np.concatenate([meta_q_drop_mean, emb_q_drop_mean])
    flip_all    = np.concatenate([meta_flip_rate,   emb_flip_rate])

    # Sort by action flip rate (most impactful first)
    sort_idx    = np.argsort(flip_all)
    labels      = [labels[i] for i in sort_idx]
    q_drop_all  = q_drop_all[sort_idx]
    flip_all    = flip_all[sort_idx]

    y      = np.arange(len(labels))
    height = 0.4

    os.makedirs("feature_importance", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Plot 1: Action Flip Rate ---
    axes[0].barh(y, flip_all, height=height * 1.5, color="steelblue")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Action Flip Rate")
    axes[0].set_title("Feature Importance — Action Flip Rate\n(how often ablation changes the decision)")
    axes[0].grid(True, axis="x")

    # Separator line between metadata and emb groups
    meta_count = len(metadata_feature_names)
    sep_pos    = (sort_idx < meta_count).sum() - 0.5  # how many meta features ended up in lower portion
    axes[0].axhline(y=sep_pos, color="red", linestyle="--", linewidth=1, alpha=0.6, label="meta / emb boundary")
    axes[0].legend(fontsize=8)

    # --- Plot 2: Q-Value Drop ---
    axes[1].barh(y, q_drop_all, height=height * 1.5, color="darkorange")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Mean Q-Value Drop")
    axes[1].set_title("Feature Importance — Q-Value Drop\n(how much expected value degrades)")
    axes[1].grid(True, axis="x")
    axes[1].axhline(y=sep_pos, color="red", linestyle="--", linewidth=1, alpha=0.6, label="meta / emb boundary")
    axes[1].legend(fontsize=8)

    plt.suptitle(
        f"Ablation Feature Importance  (n={total_states} states, {sample_n} queries)\n"
        f"Metadata ablated per-feature | Embeddings ablated per-group",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("feature_importance/ablation_importance.png", dpi=150)
    plt.close()

    # --- Console summary ---
    print(f"\n=== Ablation Summary ({total_states} states) ===")
    print(f"{'Feature':<45} {'Flip Rate':>10} {'Q-Drop':>10}")
    print("-" * 67)
    for lbl, flip, qdrop in zip(labels[::-1], flip_all[::-1], q_drop_all[::-1]):
        tag = "[META]" if lbl in metadata_feature_names else "[EMB] "
        print(f"{tag} {lbl:<38} {flip:>10.3f} {qdrop:>10.4f}")


def compute_stream_feature_importance(online_net, data, query_ids, device, max_exp_loops, n_samples=1000):
    import os
    online_net.eval()

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
        truncated = False

        while not done and not truncated:
            state_emb_var = state_emb.clone().detach().unsqueeze(0).requires_grad_(True)
            state_meta_var = state_meta.clone().detach().unsqueeze(0).requires_grad_(True)

            # Fresh forward pass
            q_vals, v_stream, a_stream = online_net(
                state_emb_var, state_meta_var, return_streams=True
            )

            action = q_vals.argmax().item()

            # --- Value stream saliency ---
            v_scalar = v_stream.sum()  # V is already scalar-like

            online_net.zero_grad()
            if state_emb_var.grad is not None:
                state_emb_var.grad.zero_()
            if state_meta_var.grad is not None:
                state_meta_var.grad.zero_()

            v_scalar.backward(retain_graph=True)

            emb_grad_v = state_emb_var.grad.detach().squeeze(0).abs().cpu()

            # --- Advantage stream saliency (selected action only) ---
            a_scalar = a_stream[0, action]  # backprop through chosen action's advantage

            online_net.zero_grad()
            state_emb_var.grad.zero_()
            state_meta_var.grad.zero_()

            a_scalar.backward()

            emb_grad_a = state_emb_var.grad.detach().squeeze(0).abs().cpu()

            # --- Accumulate ---
            if emb_sum_v is None:
                emb_sum_v = emb_grad_v.clone()
                emb_sum_a = emb_grad_a.clone()
            else:
                emb_sum_v += emb_grad_v
                emb_sum_a += emb_grad_a

            total_states += 1

            # --- Step environment ---
            if action == 0:
                next_emb, next_meta, reward, done, truncated = topic.skip()
            elif action == 1:
                next_emb, next_meta, reward, done, truncated = topic.take_single()
            elif action == 2:
                next_emb, next_meta, reward, done, truncated = topic.take_double()
            elif action == 3:
                next_emb, next_meta, reward, done, truncated = topic.take_prev_double()
            elif action == 4:
                next_emb, next_meta, reward, done, truncated = topic.take_triple()

            state_emb = next_emb.to(device)
            state_meta = next_meta.to(device)

    if total_states == 0:
        return

    emb_mean_v = emb_sum_v / total_states
    emb_mean_a = emb_sum_a / total_states

    emb_len = emb_mean_v.numel()
    groups = len(emb_group_names)

    assert emb_len % groups == 0, (
        f"Embedding length {emb_len} is not divisible by number of groups {groups}. "
        f"Check emb_group_names or embedding shape."
    )

    group_size = emb_len // groups
    emb_mean_v_groups = emb_mean_v.view(groups, group_size).mean(dim=1).numpy()
    emb_mean_a_groups = emb_mean_a.view(groups, group_size).mean(dim=1).numpy()

    labels = emb_group_names
    value_scores = emb_mean_v_groups
    adv_scores =  emb_mean_a_groups

    sort_idx = np.argsort(value_scores)
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
    val_recall = 0
    val_precision = 0
    results = []
    a2f = 0
    a2b = 0
    a3 = 0
    a1 = 0
    s = 0

    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        page = data.get_page_chunks_dict(page_id)
        query_emb = torch.tensor(query.get("query")).to(device)
        relevant_chunks = query.get("relevant_chunks")
        ranked_chunks = data.cosine_sim_rank[str(query_id)]
        query_desc = query.get("query_desc")

        logging.info(f"Query: {query_desc}")

        topic = Topic(query_emb, page, ranked_chunks, relevant_chunks, max_exp_loops)

        state_emb, state_meta, _, _ = topic.get_initial_step()
        state_emb = state_emb.to(device)
        state_meta = state_meta.to(device)
        episode_reward = 0
        done = False
        truncated = False
        episode_steps = 0

        # Greedy evaluation (no exploration)
        while not done and not truncated:
            episode_steps += 1

            with torch.no_grad():
                q = online_net(state_emb.unsqueeze(0), state_meta.unsqueeze(0))
                action = q.argmax().item()

                action_log = f"Chunk: {topic.current_chunk_id}"

                if action == 0:
                    next_emb, next_meta, reward, done, truncated = topic.skip()
                    #s += 1
                elif action == 1:
                    next_emb, next_meta, reward, done, truncated= topic.take_single()
                    #a1 += 1
                elif action == 2:
                    next_emb, next_meta, reward, done, truncated = topic.take_double()
                    #a2f += 1
                elif action == 3:
                    next_emb, next_meta, reward, done, truncated = topic.take_prev_double()
                    #a2b += 1
                elif action == 4:
                    next_emb, next_meta, reward, done, truncated = topic.take_triple()
                    #a3 += 1

                logging.info(f"{action_log}, Action Code: {action}, Reward: {reward:.4f}")

            next_emb = next_emb.to(device)
            next_meta = next_meta.to(device)
            episode_reward += reward

            state_emb, state_meta = next_emb, next_meta

        val_reward += episode_reward
        val_f1_score += topic.f1_score
        val_recall += topic.recall
        val_precision += topic.precision

        results.append({
            "query" : query_desc,
            "rl_f1_score" : topic.f1_score,
            "rl_model_retrieved" : sorted(topic.bag_of_chunks)

        })

        logging.info(f"Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}")
    
    avg_val_reward = val_reward / len(query_ids)
    avg_val_f1_score = val_f1_score / len(query_ids)
    avg_val_recall = val_recall / len(query_ids)
    avg_val_precision = val_precision / len(query_ids)

    return avg_val_reward, avg_val_f1_score, avg_val_recall, avg_val_precision, results


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

    model = DuelingDQN(metadata_dim = 6, action_dim = 5, proj_dim = 512, dropout_p = 0).to(device)
    model.load_state_dict(torch.load("models/rl-chunk-retriever_BEEEEEST22.pt", map_location="cpu")) #rl-chunk-retriever copy
    model.eval()

    avg_val_reward, avg_val_f1_score, avg_val_recall, avg_val_precision, results = evaluate(
        data_test, data_test.query_ids, model, device, 
        max_exp_loops = 1
    )
    logging.info(f"GREEDY: TEST Reward: {avg_val_reward:.4f}, TEST F1: {avg_val_f1_score:.4f}")
    print(f"GREEDY: TEST - Reward: {avg_val_reward:.4f}, F1: {avg_val_f1_score:.4f}")

    # output_dir = "data_5_analysis"
    # os.makedirs(output_dir, exist_ok=True)

    # output_file = os.path.join(output_dir, "rl_model_retrieved_test_single.json")
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)

        #############

    # pages_path = "data_3_embed/pages_chunked_emb_train.json"
    # relevant_path = "data_3_embed/relevant_chunks_emb_train.json"
    # cosine_sim_path = "data_4_cos_sim/cosine_sim_rank_threshold_only_single_train.json" #_train

    # data_train = Data(pages_path, relevant_path, cosine_sim_path)
    # data_train.load_pages()
    # data_train.load_relevant()
    # data_train.load_cosine_sim()

    # compute_stream_feature_importance(
    #     model, data_test, data_test.query_ids, device, max_exp_loops = 1, n_samples=100 #  + validation_set
    # )

    # compute_stream_ablation_importance(
    #     model, data_test, data_test.query_ids, device, max_exp_loops = 1, n_samples=100 #  + validation_set
    # )