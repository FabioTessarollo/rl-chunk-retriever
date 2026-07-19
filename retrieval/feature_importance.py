import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retrieval.Topic import Topic


def compute_stream_ablation_importance(online_net, data, query_ids, device, max_exp_loops, n_samples=1000):
    online_net.eval()

    metadata_feature_names = [
        "Rank Position",
        "Bag of Chunks Size",
        "Query - Current Chunk Sim", "Query - Current Chunk and Next Sim", "Query - Bag Sim", "Query - Current Chunk and Prev Sim"
    ]
    emb_group_names = ["Emb - Current", "Emb - Current & Next", "Emb - Current & Prev", "Emb - Query", "Emb - Bag"]

    sample_n = min(len(query_ids), n_samples)
    sampled_qids = random.sample(list(query_ids), sample_n)

    n_meta = len(metadata_feature_names)
    n_emb_groups = len(emb_group_names)

    meta_flip = np.zeros(n_meta)
    emb_flip  = np.zeros(n_emb_groups)
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
                q_base, _, _ = online_net(
                    state_emb.unsqueeze(0), state_meta.unsqueeze(0), return_streams=True
                )
                q_base_vals = q_base.squeeze(0)
                action_base = q_base_vals.argmax().item()

                for i in range(n_meta):
                    meta_ablated = state_meta.clone()
                    meta_ablated[i] = 0.0
                    q_abl, _, _ = online_net(
                        state_emb.unsqueeze(0), meta_ablated.unsqueeze(0), return_streams=True
                    )
                    action_abl = q_abl.squeeze(0).argmax().item()
                    meta_flip[i] += int(action_abl != action_base)

                emb_len = state_emb.numel()
                groups  = n_emb_groups
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
                    action_abl = q_abl.squeeze(0).argmax().item()
                    emb_flip[i] += int(action_abl != action_base)

            total_states += 1

            next_emb, next_meta, reward, done, truncated = topic.step(action_base)
            state_emb  = next_emb.to(device)
            state_meta = next_meta.to(device)

    if total_states == 0:
        return

    meta_flip_rate = meta_flip / total_states
    emb_flip_rate  = emb_flip  / total_states

    labels   = metadata_feature_names + emb_group_names
    flip_all = np.concatenate([meta_flip_rate, emb_flip_rate])

    sort_idx = np.argsort(flip_all)
    labels   = [labels[i] for i in sort_idx]
    flip_all = flip_all[sort_idx]

    colors = ["steelblue"] * len(labels)
    y = np.arange(len(labels))

    os.makedirs("feature_importance", exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.barh(y, flip_all, height=0.6, color=colors)
    plt.yticks(y, labels)
    plt.xlabel("Action Flip Rate")
    plt.title(
        f"Feature Importance — Action Flip Rate\n"
        f"(n={total_states} states, {sample_n} queries)"
    )
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig("ablation_importance.png", dpi=150)
    plt.close()

    print(f"\n=== Ablation Summary ({total_states} states) ===")
    print(f"{'Feature':<45} {'Flip Rate':>10}")
    print("-" * 57)
    for lbl, flip in zip(labels[::-1], flip_all[::-1]):
        tag = "[META]" if lbl in metadata_feature_names else "[EMB] "
        print(f"{tag} {lbl:<38} {flip:>10.3f}")


def compute_stream_feature_importance(online_net, data, query_ids, device, max_exp_loops, n_samples=1000):
    online_net.eval()

    emb_group_names = ["Emb - Current", "Emb - Current & Next", "Emb - Current & Prev", "Emb - Query", "Emb - Bag"]

    sample_n = min(len(query_ids), n_samples)
    sampled_qids = random.sample(list(query_ids), sample_n)

    emb_sum_v = None
    emb_sum_a = None
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

            q_vals, v_stream, a_stream = online_net(
                state_emb_var, state_meta_var, return_streams=True
            )

            action = q_vals.argmax().item()

            v_scalar = v_stream.sum()

            online_net.zero_grad()
            if state_emb_var.grad is not None:
                state_emb_var.grad.zero_()
            if state_meta_var.grad is not None:
                state_meta_var.grad.zero_()

            v_scalar.backward(retain_graph=True)

            emb_grad_v = state_emb_var.grad.detach().squeeze(0).abs().cpu()

            a_scalar = a_stream[0, action]

            online_net.zero_grad()
            state_emb_var.grad.zero_()
            state_meta_var.grad.zero_()

            a_scalar.backward()

            emb_grad_a = state_emb_var.grad.detach().squeeze(0).abs().cpu()

            if emb_sum_v is None:
                emb_sum_v = emb_grad_v.clone()
                emb_sum_a = emb_grad_a.clone()
            else:
                emb_sum_v += emb_grad_v
                emb_sum_a += emb_grad_a

            total_states += 1

            next_emb, next_meta, reward, done, truncated = topic.step(action)
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
