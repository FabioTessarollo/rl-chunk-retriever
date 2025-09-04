import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, embedding_dim, metadata_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.embedding_fc1 = nn.Linear(embedding_dim, 512)
        self.embedding_fc2 = nn.Linear(512, 256)
        self.combined_fc = nn.Linear(256 + metadata_dim, 256)
        self.value_fc = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)
        self.adv_fc = nn.Linear(256, 128)
        self.adv_out = nn.Linear(128, action_dim)

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