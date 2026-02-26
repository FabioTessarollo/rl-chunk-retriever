import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=256, dropout_p=0.0):
        super(DuelingDQN, self).__init__()

        self.single_proj = nn.Linear(768, proj_dim)
        self.double_proj = nn.Linear(768*2, proj_dim)
        self.prev_double_proj = nn.Linear(768*2, proj_dim)
        self.query_proj = nn.Linear(768, proj_dim)
        self.bag_proj   = nn.Linear(768, proj_dim)

        combined_dim = 5 * proj_dim 

        # shared trunk
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256  + metadata_dim, 128)
        self.dropout = nn.Dropout(p=dropout_p)

        # value stream
        self.value_fc = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

        # advantage stream
        self.adv_fc = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64, action_dim)

    def forward(self, state_embedding, state_metadata, return_streams=False):
        # split concatenated embeddings
        current, next, prev, query, bag = torch.split(state_embedding, 768, dim=-1)
        current_and_next = torch.cat([current, next], dim=-1)
        current_and_prev = torch.cat([prev, current], dim=-1)

        # project each separately
        c = F.relu(self.single_proj(current))
        cn = F.relu(self.double_proj(current_and_next))
        cp = F.relu(self.prev_double_proj(current_and_prev))
        q = F.relu(self.query_proj(query))
        b = F.relu(self.bag_proj(bag))

        # concatenate projected embeddings
        x = torch.cat([c, cn, cp, q, b], dim=-1)

        # shared trunk
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, state_metadata), dim=-1)
        x = F.relu(self.fc2(x))

        # value stream
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)

        # advantage stream
        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)

        # dueling combine
        q = v + a - a.mean(dim=-1, keepdim=True)

        if return_streams:
            return q, v, a
        return q