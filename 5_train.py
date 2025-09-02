import random
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import logging
from exp_setup import Data, Topic

class DuelingDQN(nn.Module):
    def __init__(self, embedding_dim, metadata_dim, hidden_dim=256, action_dim=6):
        super(DuelingDQN, self).__init__()
        self.embedding_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.embedding_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.combined_fc = nn.Linear(hidden_dim + metadata_dim, hidden_dim)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)
        self.adv_fc = nn.Linear(hidden_dim, hidden_dim)
        self.adv_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_embedding, state_metadata):
        x = F.relu(self.embedding_fc1(state_embedding))
        x = F.relu(self.embedding_fc2(x))
        x = torch.cat((x, state_metadata), dim=-1)
        x = F.relu(self.combined_fc(x))
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)
        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_emb, state_meta, action, reward, next_emb, next_meta, done):
        self.buffer.append((state_emb, state_meta, action, reward, next_emb, next_meta, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

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

    embedding_dim = 2304
    metadata_dim = 6

    # Hyperparameters
    gamma = 0.98
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    replay_capacity = 10000
    lr = 0.0001
    target_update = 500 #100
    epochs = 50
    max_steps_per_episode = 200 #100
    action_dim = 6

    # Set device to MPS if available (for Apple Silicon acceleration), else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    online_net = DuelingDQN(embedding_dim, metadata_dim, hidden_dim=256, action_dim=action_dim).to(device)
    target_net = DuelingDQN(embedding_dim, metadata_dim, hidden_dim=256, action_dim=action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity)

    logging.basicConfig(filename='rl_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    step_count = 0
    for epoch in range(epochs):
        epoch_reward = 0
        epoch_f1_score = 0
        random.shuffle(data.query_ids)
        print(f"Epoch: {epoch}")
        for query_id in data.query_ids:
            query = data.get_query_obj_from_id(query_id)
            page_id = query.get("page_id")
            page, page_even, page_odd = data.get_page_chunks_dict(page_id)
            query_emb = torch.tensor(query.get("query")).to(device)
            query_desc = query.get("query_desc")
            relevant_chunks = list(set(query.get("relevant_chunks")))
            ranked_chunks = data.cosine_sim_rank[str(query_id)]

            topic = Topic(query_emb, page, page_even, page_odd, ranked_chunks, relevant_chunks)

            state_emb, state_meta, _, _ = topic.get_initial_step()
            state_emb = state_emb.to(device)
            state_meta = state_meta.to(device)
            episode_reward = 0
            done = False
            episode_steps = 0
            while not done and episode_steps < max_steps_per_episode:
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
                if action == 0:
                    next_emb, next_meta, reward, done = topic.get_next_chunk_in_rank()
                elif action == 1:
                    next_emb, next_meta, reward, done = topic.get_prev_chunk_in_rank()
                elif action == 2:
                    next_emb, next_meta, reward, done = topic.get_fw_extended_chunk()
                elif action == 3:
                    next_emb, next_meta, reward, done = topic.get_bw_extended_chunk()
                elif action == 4:
                    next_emb, next_meta, reward, done = topic.add_chunk_to_bag()
                elif action == 5:
                    next_emb, next_meta, reward, done = topic.submit_current_bag()
                next_emb = next_emb.to(device)
                next_meta = next_meta.to(device)
                episode_reward += reward
                replay.push(state_emb.cpu(), state_meta.cpu(), action, reward, next_emb.cpu(), next_meta.cpu(), done)  # Store on CPU to save GPU memory
                state_emb, state_meta = next_emb, next_meta
                if len(replay) > batch_size:
                    batch = replay.sample(batch_size)
                    state_embs, state_metas, actions, rewards, next_embs, next_metas, dones = zip(*batch)
                    state_embs = torch.stack(state_embs).to(device)
                    state_metas = torch.stack(state_metas).to(device)
                    actions = torch.tensor(actions).to(device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_embs = torch.stack(next_embs).to(device)
                    next_metas = torch.stack(next_metas).to(device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(device)
                    q_values = online_net(state_embs, state_metas).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = online_net(next_embs, next_metas).argmax(1)
                        next_q = target_net(next_embs, next_metas).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        targets = rewards + gamma * next_q * (1 - dones)
                    loss = F.smooth_l1_loss(q_values, targets) #loss = F.mse_loss(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if not rand:
                        logging.info(f"Epoch: {epoch}, Query: {query_id}, Step: {episode_steps}, Loss: {loss.item():.4f}, Reward: {reward:.4f}, Action: {topic.last_action}")
                if step_count % target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    logging.info(f"Target network updated at global step {step_count}")
            epoch_reward += episode_reward
            epoch_f1_score += topic.f1_score
            logging.info(f"Epoch: {epoch}, Query: {query_desc}, Episode Reward: {episode_reward:.4f}, Episode F1: {topic.f1_score:.4f}, Bag: {topic.bag_of_chunks}, Relevant: {topic.relevant_chunks}, Top 10 Rank: {topic.ranked_chunks[:10]}, Actions: {topic.actions_taken}")
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        avg_epoch_reward = epoch_reward / len(data.query_ids)
        avg_epoch_f1_score = epoch_f1_score / len(data.query_ids)
        logging.info(f"Epoch: {epoch}, Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}, Epsilon: {epsilon:.4f}")
        print(f"Average Reward: {avg_epoch_reward:.4f}, Average F1: {avg_epoch_f1_score:.4f}")
if __name__ == "__main__":
    main()