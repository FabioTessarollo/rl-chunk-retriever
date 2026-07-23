import logging
from dataclasses import dataclass, field

import torch

from retrieval.environment import Topic


@dataclass
class EvalResult:
    avg_reward: float = 0
    avg_f1: float = 0
    avg_recall: float = 0
    avg_precision: float = 0
    action_counts: dict = field(default_factory=dict)
    probs: list = field(default_factory=list)
    results: list = field(default_factory=list)


def evaluate(data, query_ids, model, device, max_exp_loops,
             reward_cfg=None, track_actions=False, collect_results=False,
             history=None, epoch=None):
    """Unified evaluation: greedy policy over query_ids.

    Args:
        track_actions: If True, counts actions per type and computes exploration probs.
        collect_results: If True, collects per-query results for JSON output.
        history: Mutable list to append action counts (used when track_actions=True).
        epoch: Current epoch number (used when track_actions=True).
    """
    val_reward = 0
    val_f1_score = 0
    val_recall = 0
    val_precision = 0
    results = []

    epoch_counts = {'skip': 0, 'take_1': 0, 'take_2n': 0, 'take_2p': 0, 'take_3': 0}
    action_keys = ['skip', 'take_1', 'take_2n', 'take_2p', 'take_3']

    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        page = data.get_page_chunks_dict(page_id)
        query_emb = torch.tensor(query.get("query")).to(device)
        relevant_chunks = query.get("relevant_chunks")
        ranked_chunks = data.cosine_sim_rank[str(query_id)]
        query_desc = query.get("query_desc")

        logging.info(f"Query: {query_desc}")

        topic = Topic(query_emb, page, ranked_chunks, relevant_chunks, max_exp_loops, device, reward_cfg)

        state_emb, state_meta, _, _ = topic.get_initial_step()
        state_emb = state_emb.to(device)
        state_meta = state_meta.to(device)
        episode_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            with torch.no_grad():
                q = model(state_emb.unsqueeze(0), state_meta.unsqueeze(0))
                action = q.argmax().item()

                logging.info(f"Chunk: {topic.current_chunk_id}, Action Code: {action}")

                next_emb, next_meta, reward, done, truncated = topic.step(action)

                if track_actions:
                    epoch_counts[action_keys[action]] += 1

            next_emb = next_emb.to(device)
            next_meta = next_meta.to(device)
            episode_reward += reward

            state_emb, state_meta = next_emb, next_meta

        val_reward += episode_reward
        val_f1_score += topic.f1_score
        val_recall += topic.recall
        val_precision += topic.precision

        if collect_results:
            results.append({
                "query": query_desc,
                "rl_f1_score": topic.f1_score,
                "rl_model_retrieved": sorted(topic.bag_of_chunks)
            })

        logging.info(f"GREEDY - Query: {query_desc}, Episode Reward: {episode_reward:.4f}, "
                     f"Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, "
                     f"Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}")

    n = len(query_ids)
    result = EvalResult(
        avg_reward=val_reward / n,
        avg_f1=val_f1_score / n,
        avg_recall=val_recall / n,
        avg_precision=val_precision / n,
    )

    if track_actions:
        if epoch is not None:
            epoch_counts['epoch'] = epoch
        if history is not None:
            history.append(epoch_counts)
        result.action_counts = epoch_counts

        actions_f = [epoch_counts[k] for k in action_keys]
        first_value = sorted(actions_f)[0]
        second_value = sorted(actions_f)[1]
        if second_value + first_value < 50:
            weights = [1 / (f + 500) for f in actions_f]
            total = sum(weights)
            result.probs = [w / total for w in weights]

    if collect_results:
        result.results = results

    return result
