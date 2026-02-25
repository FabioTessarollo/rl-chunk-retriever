import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from datetime import datetime
import argparse
import matplotlib
matplotlib.use('Agg') # Must be called before importing plt
import matplotlib.pyplot as plt
import numpy as np

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
        episode_steps = 0
        
        # Greedy evaluation (no exploration)
        while not done:
            episode_steps += 1
            
            with torch.no_grad():
                q = online_net(state_emb.unsqueeze(0), state_meta.unsqueeze(0))
                action = q.argmax().item()
            
                action_log = f"Chunk: {topic.current_chunk_id}"
                
                if topic.current_loop + 1 > max_exp_loops:
                    action = 4
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

                logging.info(f"{action_log}, Action Code: {action}, Reward: {reward:.4f}")
                
            next_emb = next_emb.to(device)
            next_meta = next_meta.to(device)
            episode_reward += reward
            
            state_emb, state_meta = next_emb, next_meta
            
        val_reward += episode_reward
        val_f1_score += topic.f1_score

        logging.info(f"GREEDY - Query: {query_desc}, Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}")
    
    avg_val_reward = val_reward / len(query_ids)
    avg_val_f1_score = val_f1_score / len(query_ids)
    
    return avg_val_reward, avg_val_f1_score

def train():

    pages_path = "data_3_embed/pages_chunked_emb_train.json"
    relevant_path = "data_3_embed/relevant_chunks_emb_train.json"
    cosine_sim_path = "data_4_cos_sim/cosine_sim_rank_threshold_only_single_train.json" #_train

    data = Data(pages_path, relevant_path, cosine_sim_path)
    data.load_pages()
    data.load_relevant()
    data.load_cosine_sim()

    train_set, validation_set = data.balanced_split_query_ids(data.query_ids, 0.6)

    best_score = 0
    metadata_dim = 9
    epsilon = 1.0

    proj_dim = 512
    gamma = 0.99
    epsilon_min = 0.01
    epsilon_decay = 0.99995
    batch_size = 32
    replay_capacity = 50000
    lr = 2e-5
    target_update = 2000 ############### PROVARE A DIMINUIRE
    epochs = 60# 31 #24
    max_exp_loops = 1
    action_dim = 5
    dropout_p = 0
    scheduler_type = "cosine"
    per_alpha = 0.6
    per_beta = 0.4
    per_beta_increment = 0.001
    eta_min = 0.000001
    warm_up_epoches = 30
    neg_schedule = torch.linspace(0.2, 1.0, steps=warm_up_epoches)


    es = EarlyStopping(patience=10, delta_ratio=0.001) #12? #15? #10?

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    online_net = DuelingDQN(metadata_dim, action_dim, proj_dim, dropout_p).to(device)
    target_net = DuelingDQN(metadata_dim, action_dim, proj_dim, dropout_p).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)#, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=eta_min)
      
    replay = PrioritizedReplayBuffer(
        capacity=replay_capacity, 
        alpha=per_alpha, 
        beta=per_beta, 
        beta_increment=per_beta_increment
    )

    logging.basicConfig(filename='rl_training.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode="w")
    extra_logger = logging.getLogger("extra")
    extra_handler = logging.FileHandler("hyperparameters.csv", mode="a")
    extra_logger.addHandler(extra_handler)
    extra_logger.setLevel(logging.INFO)

    step_count = 0
    train_f1_scores = []
    val_f1_scores = []
    for epoch in range(epochs):
        online_net.train()
        epoch_reward = 0
        epoch_f1_score = 0
        random.shuffle(train_set)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        logging.info(f"Epoch {epoch} - Learning Rate: {current_lr:.6f}")

        if epoch < warm_up_epoches:
            p = neg_schedule[epoch].item()

        print(f"Epoch: {epoch}")
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
            if epoch <= warm_up_epoches and not any(item in ranked_chunks for item in relevant_chunks):
                continue

            logging.info(f"Epoch: {epoch}, Query: {query_desc}")

            topic = Topic(query_emb, page, ranked_chunks, relevant_chunks, max_exp_loops)

            state_emb, state_meta, _, _ = topic.get_initial_step()
            state_emb = state_emb.to(device)
            state_meta = state_meta.to(device)
            episode_reward = 0
            done = False
            episode_steps = 0

            while not done:
                episode_steps += 1
                step_count += 1
                if random.random() < epsilon:
                    rand = True
                    action = random.randint(0, action_dim - 1)
                else:
                    rand = False
                    with torch.no_grad():
                        q = online_net(state_emb.unsqueeze(0), state_meta.unsqueeze(0))
                        action = q.argmax().item()
                
                action_log = f"Chunk: {topic.current_chunk_id}"
                
                if topic.current_loop + 1 > max_exp_loops:
                    action = 4
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

                logging.info(f"{action_log}, Action Code: {action}, Reward: {reward:.4f}, Rand: {int(rand)}")
                    
                next_emb = next_emb.to(device)
                next_meta = next_meta.to(device)
                episode_reward += reward
                
                # Store on CPU to save GPU memory
                replay.push(
                    state_emb.cpu(), state_meta.cpu(), action, reward, 
                    next_emb.cpu(), next_meta.cpu(), done
                )
                
                state_emb, state_meta = next_emb, next_meta
                
                if len(replay) > batch_size:
                    # Sample from PER buffer
                    batch, idxs, is_weights = replay.sample(batch_size)
                    
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
                        targets = rewards + gamma * next_q * (1 - dones)
                    
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
                
                if step_count % target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    logging.info(f"Target network updated at global step {step_count}")
                    
            epoch_reward += episode_reward
            epoch_f1_score += topic.f1_score
            logging.info(f"Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}")
            
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                
        avg_epoch_reward = epoch_reward / len(train_set)
        avg_epoch_f1_score = epoch_f1_score / len(train_set)

        logging.info(f"Epoch: {epoch}, Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}, Epsilon: {epsilon:.4f}")
        print(f"Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}")

        scheduler.step()

        
        # Greedy evaluation
        if epoch > 40:
            online_net.eval()

            avg_train_reward, avg_train_f1_score = evaluate(
                data, train_set, online_net, device, 
                max_exp_loops
            )
            logging.info(f"GREEDY: Train Reward: {avg_train_reward:.4f}, Val F1: {avg_train_f1_score:.4f}")
            print(f"GREEDY: Train - Reward: {avg_train_reward:.4f}, F1: {avg_train_f1_score:.4f}")
            train_f1_scores.append(avg_train_f1_score)

            avg_val_reward, avg_val_f1_score = evaluate(
                data, validation_set, online_net, device, 
                max_exp_loops
            )
            logging.info(f"GREEDY: Val Reward: {avg_val_reward:.4f}, Val F1: {avg_val_f1_score:.4f}")
            print(f"GREEDY: Validation - Reward: {avg_val_reward:.4f}, F1: {avg_val_f1_score:.4f}")
            val_f1_scores.append(avg_val_f1_score)

        #     if es.step(avg_val_f1_score):
        #         print(f"Early stopping at epoch {epoch}")
        #         break

    extra_logger.info(f"{now_str}\t{proj_dim}\t{gamma}\t{epsilon_min}\t{epsilon_decay}\t{batch_size}\t{replay_capacity}\t{lr}\t{target_update}\t{epochs}\t{max_exp_loops}\t{action_dim}\t{scheduler_type}\t{per_alpha}\t{per_beta}\t{per_beta_increment}\t{dropout_p}\t{best_score:.4f}")
    
    # plt.figure(figsize=(7,5))
    # plt.plot(train_f1_scores, label='train')
    # plt.plot(val_f1_scores, label='val')
    # plt.legend()
    # plt.xlabel('Epoches')
    # plt.ylabel('F1 Score')
    # plt.title('Train and Validation F1 Scores')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('train_vs_val.png')

    # trained_model_path = "models/rl-chunk-retriever.pt"
    # torch.save(online_net.state_dict(), trained_model_path)

