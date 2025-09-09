import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=128):
        super(DuelingDQN, self).__init__()

        self.single_proj = nn.Linear(768, proj_dim)
        self.double_proj = nn.Linear(768, proj_dim)
        self.query_proj = nn.Linear(768, proj_dim)
        self.bag_proj   = nn.Linear(768, proj_dim)

        combined_dim = 4 * proj_dim  # 512 if proj_dim=128

        # shared trunk
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 128)

        # value stream
        self.value_fc = nn.Linear(128, 64)
        self.value_out = nn.Linear(64 + metadata_dim, 1)

        # advantage stream
        self.adv_fc = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64 + metadata_dim, action_dim)

    def forward(self, state_embedding, state_metadata):
        # split concatenated embeddings: (batch, 3072) → 4 × (batch, 768)
        single, double, query, bag = torch.split(state_embedding, 768, dim=-1)

        # project each separately
        s = F.relu(self.single_proj(single))
        d = F.relu(self.double_proj(double))
        q = F.relu(self.query_proj(query))
        b = F.relu(self.bag_proj(bag))

        # concatenate projected embeddings
        x = torch.cat([s, d, q, b], dim=-1)

        # shared trunk
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # value stream
        v = F.relu(self.value_fc(x))
        v = torch.cat((v, state_metadata), dim=-1)
        v = self.value_out(v)

        # advantage stream
        a = F.relu(self.adv_fc(x))
        a = torch.cat((a, state_metadata), dim=-1)
        a = self.adv_out(a)

        # dueling combine
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
