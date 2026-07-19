import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from datetime import datetime
import numpy as np

from retrieval.data_loader import Data
from retrieval.environment import Topic
from retrieval.dueling_dqn import DuelingDQN
from retrieval.replay_buffer import PrioritizedReplayBuffer
from retrieval.evaluate import evaluate
from retrieval.plotting import plot_train_val_metric, plot_action_counts
from config import get_config, get_device, set_seed

logger = logging.getLogger(__name__)
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train(cfg=None):
    if cfg is None:
        cfg = get_config()

    set_seed(cfg)
    device = get_device(cfg)
    logger.info(f"Using device: {device}")

    embed_dir = cfg.data.embed_dir
    cos_sim_dir = cfg.data.cos_sim_dir
    t = cfg.training
    reward_cfg = cfg.retrieval.reward

    pages_path = f"{embed_dir}/pages_chunked_emb_train.json"
    relevant_path = f"{embed_dir}/relevant_chunks_emb_train.json"
    cosine_sim_path = f"{cos_sim_dir}/cosine_sim_rank_threshold_only_single_train.json"

    data = Data(pages_path, relevant_path, cosine_sim_path, device)
    data.load_pages()
    data.load_relevant()
    data.load_cosine_sim()

    train_set, validation_set = data.balanced_split_query_ids(data.query_ids, t.train_split)

    epsilon = t.epsilon

    neg_schedule = torch.linspace(t.neg_schedule_start, t.neg_schedule_end, steps=t.neg_schedule_steps)
    probs = []
    history = []

    online_net = DuelingDQN(t.metadata_dim, t.action_dim, t.proj_dim, t.dropout, cfg.model.embedding_dim).to(device)
    target_net = DuelingDQN(t.metadata_dim, t.action_dim, t.proj_dim, t.dropout, cfg.model.embedding_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=t.lr, weight_decay=t.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.scheduler_t_max, eta_min=t.eta_min)

    replay = PrioritizedReplayBuffer(
        capacity=t.replay_capacity,
        alpha=t.per_alpha,
        beta=t.per_beta,
        beta_increment=t.per_beta_increment
    )

    extra_logger = logging.getLogger("extra")
    extra_handler = logging.FileHandler("hyperparameters.csv", mode="a")
    extra_logger.addHandler(extra_handler)
    extra_logger.setLevel(logging.INFO)

    step_count = 0
    train_f1_scores = []
    val_f1_scores = []
    train_rewards = []
    val_rewards = []
    train_recalls = []
    val_recalls = []
    for epoch in range(t.epochs):
        online_net.train()
        epoch_reward = 0
        epoch_f1_score = 0
        random.shuffle(train_set)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} - Learning Rate: {current_lr:.6f}")

        if epoch < t.warm_up_epochs:
            p = neg_schedule[epoch].item()

        logger.info(f"Epoch: {epoch}")
        for query_id in train_set:
            query = data.get_query_obj_from_id(query_id)
            page_id = query.get("page_id")
            page = data.get_page_chunks_dict(page_id)
            query_emb = torch.tensor(query.get("query")).to(device)
            query_desc = query.get("query_desc")
            relevant_chunks = query.get("relevant_chunks")
            ranked_chunks = data.cosine_sim_rank[str(query_id)] #ranked_chunks = data.get_ranked_with_prev_chunks_from_query_id(query_id)


            ranked_chunks = [
                c for c in ranked_chunks
                if (c in relevant_chunks) or (torch.rand(1).item() < p)
            ]
            if not any(item in ranked_chunks for item in relevant_chunks):
                continue

            logging.info(f"Epoch: {epoch}, Query: {query_desc}")

            topic = Topic(query_emb, page, ranked_chunks, relevant_chunks, t.max_exp_loops, device, reward_cfg)

            state_emb, state_meta, _, _ = topic.get_initial_step()
            state_emb = state_emb.to(device)
            state_meta = state_meta.to(device)
            episode_reward = 0
            done = False
            truncated = False
            episode_steps = 0

            while not done and not truncated:
                episode_steps += 1
                step_count += 1
                if random.random() < epsilon:
                    if probs:
                        action = random.choices(range(t.action_dim), weights=probs, k=1)[0]
                    else:
                        action = random.randint(0, t.action_dim - 1)
                else:
                    with torch.no_grad():
                        q = online_net(state_emb.unsqueeze(0), state_meta.unsqueeze(0))
                        action = q.argmax().item()

                next_emb, next_meta, reward, done, truncated = topic.step(action)

                next_emb = next_emb.to(device)
                next_meta = next_meta.to(device)
                episode_reward += reward

                # Store on CPU to save GPU memory
                replay.push(
                    state_emb.cpu(), state_meta.cpu(), action, reward,
                    next_emb.cpu(), next_meta.cpu(), done
                )

                state_emb, state_meta = next_emb, next_meta

                if len(replay) > t.batch_size:
                    # Sample from PER buffer
                    batch, idxs, is_weights = replay.sample(t.batch_size)

                    state_embs, state_metas, actions, rewards, next_embs, next_metas, dones = zip(*batch)
                    state_embs = torch.stack(state_embs).to(device)
                    state_metas = torch.stack(state_metas).to(device)
                    actions = torch.tensor(actions).to(device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_embs = torch.stack(next_embs).to(device)
                    next_metas = torch.stack(next_metas).to(device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(device)
                    is_weights = torch.tensor(is_weights, dtype=torch.float32).to(device)

                    # Compute Q-values and targets
                    q_values = online_net(state_embs, state_metas).gather(1, actions.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        # Double DQN: use online network to select actions, target network to evaluate
                        next_actions = online_net(next_embs, next_metas).argmax(1)
                        next_q = target_net(next_embs, next_metas).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        targets = rewards + t.gamma * next_q * (1 - dones)

                    # Compute TD errors for priority updates
                    td_errors = (q_values - targets).detach().cpu().numpy()

                    # Apply importance sampling weights to loss
                    elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction='none')
                    loss = (is_weights * elementwise_loss).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update priorities in replay buffer
                    replay.update_priorities(idxs, td_errors)

                if step_count % t.target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    logging.info(f"Target network updated at global step {step_count}")

            epoch_reward += episode_reward
            epoch_f1_score += topic.f1_score
            logging.info(f"Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}")

            if epsilon > t.epsilon_min:
                epsilon *= t.epsilon_decay

        avg_epoch_reward = epoch_reward / len(train_set)
        avg_epoch_f1_score = epoch_f1_score / len(train_set)

        logger.info(f"Epoch: {epoch}, Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}, Epsilon: {epsilon:.4f}")

        scheduler.step()


        online_net.eval()

        train_result = evaluate(
            data, train_set, online_net, device,
            t.max_exp_loops, reward_cfg=reward_cfg, track_actions=True,
            history=history, epoch=epoch
        )
        logger.info(f"GREEDY: Train - Reward: {train_result.avg_reward:.4f}, F1: {train_result.avg_f1:.4f}, Recall {train_result.avg_recall:.4f}, Precision {train_result.avg_precision:.4f}")
        train_f1_scores.append(train_result.avg_f1)
        train_rewards.append(train_result.avg_reward)
        train_recalls.append(train_result.avg_recall)

        val_result = evaluate(
            data, validation_set, online_net, device,
            t.max_exp_loops, reward_cfg=reward_cfg, track_actions=True,
            history=history, epoch=epoch
        )
        logger.info(f"GREEDY: Val - Reward: {val_result.avg_reward:.4f}, F1: {val_result.avg_f1:.4f}, Recall {val_result.avg_recall:.4f}, Precision {val_result.avg_precision:.4f}")
        val_f1_scores.append(val_result.avg_f1)
        val_rewards.append(val_result.avg_reward)
        val_recalls.append(val_result.avg_recall)

        probs_result = evaluate(
            data, random.sample(train_set, 50), online_net, device,
            t.max_exp_loops, reward_cfg=reward_cfg, track_actions=True,
            history=history, epoch=epoch
        )
        probs = probs_result.probs



    extra_logger.info(f"{now_str}\t{t.proj_dim}\t{t.gamma}\t{t.epsilon_min}\t{t.epsilon_decay}\t{t.batch_size}\t{t.replay_capacity}\t{t.lr}\t{t.target_update}\t{t.epochs}\t{t.max_exp_loops}\t{t.action_dim}\tcosine\t{t.per_alpha}\t{t.per_beta}\t{t.per_beta_increment}\t{t.dropout}")

    plot_train_val_metric(train_f1_scores, val_f1_scores, 'F1 Score', 'train_vs_val_f1_score.png')
    plot_train_val_metric(train_rewards, val_rewards, 'Reward', 'train_vs_val_reward.png')
    plot_train_val_metric(train_recalls, val_recalls, 'Recall', 'train_vs_val_recall.png')
    plot_action_counts(history, 'train actions.png')

