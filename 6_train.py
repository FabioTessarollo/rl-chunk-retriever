import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from datetime import datetime
import argparse

from Data import Data
from Topic import Topic
from DuelingDQN import DuelingDQN
from ReplayBuffer import PrioritizedReplayBuffer
from EarlyStopping import EarlyStopping

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
random.seed(1)
torch.manual_seed(1)

def print_branch_importance(model, proj_dim):
    # ---- Branch importance (fc1) ----
    W = model.fc1.weight.detach().cpu()

    W_single = W[:, 0:proj_dim]
    W_double = W[:, proj_dim:2*proj_dim]
    W_query  = W[:, 2*proj_dim:3*proj_dim]
    W_bag    = W[:, 3*proj_dim:4*proj_dim]

    norms = {
        "single_chunk": torch.norm(W_single).item(),
        "double_chunk": torch.norm(W_double).item(),
        "query": torch.norm(W_query).item(),
        "bag": torch.norm(W_bag).item(),
    }

    total = sum(norms.values())
    rel = {k: v/total for k, v in norms.items()}

    print("Branch weight norms (absolute):")
    for k, v in norms.items():
        print(f"  {k:>12}: {v:.4f}")

    print("\nBranch relative importance (normalized):")
    for k, v in rel.items():
        print(f"  {k:>12}: {100*v:.2f}%")

    # ---- Metadata neuron importance in advantage stream ----
    W_adv = model.adv_out.weight.detach().cpu()  # shape: (action_dim, 64 + metadata_dim)

    hidden_weights = W_adv[:, :64]                     # (action_dim, 64)
    metadata_weights = W_adv[:, 64:]                   # (action_dim, metadata_dim)

    hidden_norms = torch.norm(hidden_weights, dim=0)   # (64,)
    metadata_norms = torch.norm(metadata_weights, dim=0)  # (metadata_dim,)

    # totals
    total_hidden = hidden_norms.sum().item()
    total_meta = metadata_norms.sum().item()

    # normalize metadata importances (relative only among metadata)
    rel_meta = metadata_norms / total_meta if total_meta > 0 else metadata_norms

    print("\nAdvantage stream metadata neuron importances:")
    for i, (abs_val, rel_val) in enumerate(zip(metadata_norms.tolist(), rel_meta.tolist()), 1):
        print(f"  metadata_{i:02d}: abs={abs_val:.4f}, rel={100*rel_val:.2f}%")

    print(f"\nTotal hidden neurons importance (sum abs norms): {total_hidden:.4f}")
    print(f"Average hidden neuron importance (abs norm): {hidden_norms.mean().item():.4f}")
    print(f"Total metadata neurons importance (sum abs norms): {total_meta:.4f}")


def evaluate_on_validation(data, validation_set, online_net, device, max_exp_loops):
    """Evaluate the model on validation set without training"""
    val_reward = 0
    val_f1_score = 0
    
    for query_id in validation_set:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        page, page_even, page_odd = data.get_page_chunks_dict(page_id)
        query_emb = torch.tensor(query.get("query")).to(device)
        relevant_chunks = query.get("relevant_chunks")
        ranked_chunks = data.cosine_sim_rank[str(query_id)]
        query_desc = query.get("query_desc")
        avg_similarity = data.get_avg_sim(query_id)

        topic = Topic(query_emb, page, page_even, page_odd, ranked_chunks, relevant_chunks, max_exp_loops, avg_similarity)

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
                    next_emb, next_meta, reward, done = topic.submit_current_bag()
                
            next_emb = next_emb.to(device)
            next_meta = next_meta.to(device)
            episode_reward += reward
            
            state_emb, state_meta = next_emb, next_meta

            logging.info(f"Step: {episode_steps}, Reward: {reward:.4f}, Action: {action}")
            
        val_reward += episode_reward
        val_f1_score += topic.f1_score

        logging.info(f"GREEDY - Query: {query_desc}, Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}, Actions: {topic.actions_taken}")
    
    avg_val_reward = val_reward / len(validation_set)
    avg_val_f1_score = val_f1_score / len(validation_set)
    
    return avg_val_reward, avg_val_f1_score

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

    fair_query_ids, superdifficult_query_ids = data.get_query_ids_by_difficulty()

    train_set, validation_set = data.balanced_split_query_ids(fair_query_ids, 0.7)

    best_score = 0
    proj_dim = 256
    metadata_dim = 8
    epsilon = 1.0

    # TO DO
    # try removing the / in between before embedding
    # passing metadata earlier
    # curriculum learning by epoches
    # reduce PER BETA to lower biased sampling on difficult data
    # TD3
    # starting with some weights
    # dividere la similarità per la sim media
    # provare reward negativo skip solo dopo il primo loop
    # provare a non includere il submit nelle azioni random
    # more epoches with the 0 reward for bad skip in first loop

    parser = argparse.ArgumentParser()

    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99) #0.99
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.99993)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max_exp_loops", type=int, default=3) #3
    parser.add_argument("--action_dim", type=int, default=4) #4
    parser.add_argument("--dropout_p", type=float, default=0)

    parser.add_argument("--scheduler_type", default="plateau") # Options: "step", "cosine", "exponential", "plateau"
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta", type=float, default=0.4)
    parser.add_argument("--per_beta_increment", type=float, default=0.001)

    args = parser.parse_args()

    proj_dim, gamma, epsilon_min, epsilon_decay, batch_size, replay_capacity, lr, target_update, epochs, max_exp_loops, action_dim, scheduler_type, per_alpha, per_beta, per_beta_increment, dropout_p = \
        args.proj_dim, args.gamma, args.epsilon_min, args.epsilon_decay, args.batch_size, args.replay_capacity, args.lr, args.target_update, args.epochs, args.max_exp_loops, args.action_dim, args.scheduler_type, args.per_alpha, args.per_beta, args.per_beta_increment, args.dropout_p

    # Scheduler hyperparameters
    step_size = 10  # For StepLR
    gamma_scheduler = 0.9  # For StepLR and ExponentialLR
    eta_min = 1e-5  # For CosineAnnealingLR
    patience = 5  # For ReduceLROnPlateau #5 
    factor = 0.5  # For ReduceLROnPlateau

    # PER hyperparameters
    per_alpha = 0.6      # Prioritization exponent #0.6 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    per_beta = 0.4       # Importance sampling correction
    per_beta_increment = 0.001  # Beta annealing rate

    # Early Stopping
    es = EarlyStopping(patience=30, delta_ratio=0.01)

    # Set device to MPS if available (for Apple Silicon acceleration), else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    online_net = DuelingDQN(metadata_dim, action_dim, proj_dim, dropout_p).to(device)
    target_net = DuelingDQN(metadata_dim, action_dim, proj_dim, dropout_p).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr) #, weight_decay=1e-5)

        # Initialize scheduler
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_scheduler)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-20, eta_min=eta_min)
    elif scheduler_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_scheduler)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, 
                                                       patience=patience)
      
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
    for epoch in range(epochs):
        online_net.train()
        epoch_reward = 0
        epoch_f1_score = 0
        random.shuffle(train_set)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        logging.info(f"Epoch {epoch} - Learning Rate: {current_lr:.6f}")

        print(f"Epoch: {epoch}")
        for query_id in train_set:
            query = data.get_query_obj_from_id(query_id)
            page_id = query.get("page_id")
            page, page_even, page_odd = data.get_page_chunks_dict(page_id)
            query_emb = torch.tensor(query.get("query")).to(device)
            query_desc = query.get("query_desc")
            relevant_chunks = query.get("relevant_chunks")
            ranked_chunks = data.get_ranked_with_prev_chunks_from_query_id(query_id)
            avg_similarity = data.get_avg_sim(query_id)

            topic = Topic(query_emb, page, page_even, page_odd, ranked_chunks, relevant_chunks, max_exp_loops, avg_similarity)

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
                
                if topic.current_loop + 1 > max_exp_loops:
                    next_emb, next_meta, reward, done = topic.submit_current_bag()
                elif action == 0:
                    next_emb, next_meta, reward, done = topic.skip()
                elif action == 1:
                    next_emb, next_meta, reward, done = topic.take_single()
                elif action == 2:
                    next_emb, next_meta, reward, done = topic.take_double()
                elif action == 3 and action_dim == 4:
                    next_emb, next_meta, reward, done = topic.submit_current_bag()
                    
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
                    
                    logging.info(f"Epoch: {epoch}, Query: {query_id}, Step: {episode_steps}, Loss: {loss.item():.4f}, Reward: {reward:.4f}, Rand: {int(rand)}, Action Code: {action}")
                
                if step_count % target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    logging.info(f"Target network updated at global step {step_count}")
                    
            epoch_reward += episode_reward
            epoch_f1_score += topic.f1_score
            logging.info(f"Epoch: {epoch}, Query: {query_desc}, Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top_10_Rank: {topic.ranked_chunks[:10]}, Actions: {topic.actions_taken}")
            
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                
        avg_epoch_reward = epoch_reward / len(fair_query_ids)
        avg_epoch_f1_score = epoch_f1_score / len(fair_query_ids)

        logging.info(f"Epoch: {epoch}, Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}, Epsilon: {epsilon:.4f}")
        print(f"Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}")

        # TRAIN SET GREEDY
        avg_train_reward, avg_train_f1_score = evaluate_on_validation(
            data, train_set, online_net, device, 
            max_exp_loops
        )
        logging.info(f"GREEDY: Train Reward: {avg_train_reward:.4f}, Val F1: {avg_train_f1_score:.4f}, Epsilon: {epsilon:.4f}")
        print(f"GREEDY: Train - Reward: {avg_train_reward:.4f}, F1: {avg_train_f1_score:.4f}")

        # VALIDATION SET GREEDY
        online_net.eval()
        avg_val_reward, avg_val_f1_score = evaluate_on_validation(
            data, validation_set, online_net, device, 
            max_exp_loops
        )
        logging.info(f"GREEDY: Val Reward: {avg_val_reward:.4f}, Val F1: {avg_val_f1_score:.4f}, Epsilon: {epsilon:.4f}")
        print(f"GREEDY: Validation - Reward: {avg_val_reward:.4f}, F1: {avg_val_f1_score:.4f}")

        # Step the scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(avg_val_f1_score)
            else:
                scheduler.step()

        if avg_val_f1_score > best_score:
            best_score = avg_val_f1_score
            torch.save(online_net.state_dict(), "models/rl-chunk-retriever.pt")
        
        if epoch > 2:
            if es.step(avg_val_f1_score):
                print(f"Early stopping at epoch {epoch}")
                break

    extra_logger.info(f"{now_str}\t{proj_dim}\t{gamma}\t{epsilon_min}\t{epsilon_decay}\t{batch_size}\t{replay_capacity}\t{lr}\t{target_update}\t{epochs}\t{max_exp_loops}\t{action_dim}\t{scheduler_type}\t{per_alpha}\t{per_beta}\t{per_beta_increment}\t{dropout_p}\t{best_score:.4f}")
    
    print_branch_importance(online_net, proj_dim)

if __name__ == "__main__":
    main()