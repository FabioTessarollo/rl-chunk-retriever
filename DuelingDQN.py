import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
class DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=256, dropout_p=0.0):
        super(DuelingDQN, self).__init__()

        self.single_proj = nn.Linear(768, proj_dim)
        self.double_proj = nn.Linear(768, proj_dim)
        self.query_proj = nn.Linear(768, proj_dim)
        self.bag_proj   = nn.Linear(768, proj_dim)

        combined_dim = 4 * proj_dim 

        # shared trunk
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=dropout_p)

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
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

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

"""

class DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=256, dropout_p=0.0):
        super(DuelingDQN, self).__init__()

        self.single_proj = nn.Linear(768, proj_dim)
        self.double_proj = nn.Linear(768, proj_dim)
        self.query_proj = nn.Linear(768, proj_dim)
        self.bag_proj   = nn.Linear(768, proj_dim)

        combined_dim = 4 * proj_dim 

        # shared trunk
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=dropout_p)

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
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

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



"""
class DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=128, dropout_p=0.1):
        super(DuelingDQN, self).__init__()

        self.single_proj = nn.Linear(768, proj_dim)
        self.double_proj = nn.Linear(768, proj_dim)
        self.query_proj = nn.Linear(768, proj_dim)
        self.bag_proj   = nn.Linear(768, proj_dim)

        combined_dim = 4 * proj_dim 

        # shared trunk
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=dropout_p)

        # value stream
        self.value_fc = nn.Linear(256, 128)
        self.value_out = nn.Linear(128 + metadata_dim, 1)

        # advantage stream
        self.adv_fc = nn.Linear(256, 128)
        self.adv_out = nn.Linear(128 + metadata_dim, action_dim)

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
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

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



"""



# attention
"""
lass DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=128, dropout_p=0.0, attn_heads=None):
        super(DuelingDQN, self).__init__()

        # per-branch projection
        self.single_proj = nn.Linear(768, proj_dim)
        self.double_proj = nn.Linear(768, proj_dim)
        self.query_proj = nn.Linear(768, proj_dim)
        self.bag_proj   = nn.Linear(768, proj_dim)

        # attention settings
        if attn_heads is None:
            # choose a sensible default: 4 heads if divisible, else 1
            attn_heads = 4 if (proj_dim % 4 == 0) else 1
        self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=attn_heads, batch_first=True)

        combined_dim = 4 * proj_dim  # will flatten attention outputs

        # shared trunk
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=dropout_p)

        # value stream (keep 128 hidden size)
        self.value_fc = nn.Linear(128, 128)
        self.value_out = nn.Linear(128 + metadata_dim, 1)

        # advantage stream (mirror of value)
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128 + metadata_dim, action_dim)

        # small init for final layers (helps RL stability)
        nn.init.uniform_(self.value_out.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.adv_out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.value_out.bias)
        nn.init.zeros_(self.adv_out.bias)

    def forward(self, state_embedding, state_metadata):
        # state_embedding: (batch, 4*768)
        single, double, query, bag = torch.split(state_embedding, 768, dim=-1)

        # project each branch -> (batch, proj_dim)
        s = F.relu(self.single_proj(single))
        d = F.relu(self.double_proj(double))
        q = F.relu(self.query_proj(query))
        b = F.relu(self.bag_proj(bag))

        # stack as sequence for attention: (batch, seq_len=4, proj_dim)
        seq = torch.stack([s, d, q, b], dim=1)

        # self-attention fusion (queries=keys=values=seq)
        attn_out, _ = self.attn(seq, seq, seq)  # (batch, 4, proj_dim)

        # flatten attention outputs into single vector per sample
        x = attn_out.reshape(attn_out.size(0), -1)  # (batch, 4*proj_dim)

        # shared trunk
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # value stream
        v = F.relu(self.value_fc(x))                 # (batch, 128)
        v = torch.cat((v, state_metadata), dim=-1)   # (batch, 128 + metadata_dim)
        v = self.value_out(v)                        # (batch, 1)

        # advantage stream
        a = F.relu(self.adv_fc(x))                   # (batch, 128)
        a = torch.cat((a, state_metadata), dim=-1)   # (batch, 128 + metadata_dim)
        a = self.adv_out(a)                          # (batch, action_dim)

        # dueling combine
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q


"""


# 0.2614

"""
class DuelingDQN(nn.Module):
    def __init__(self, metadata_dim, action_dim, proj_dim=128, dropout_p=0.1):
        super(DuelingDQN, self).__init__()

        embedding_size = 768

        # shared trunk
        self.fc_single = nn.Linear(embedding_size*2, 128)
        self.fc_double = nn.Linear(embedding_size*2, 128)
        self.fc_bag = nn.Linear(embedding_size*2, 128)

        self.fc2 = nn.Linear(128*3, 128)
        self.dropout = nn.Dropout(p=dropout_p)

        # value stream
        self.value_fc = nn.Linear(128, 128)
        self.value_out = nn.Linear(128 + metadata_dim, 1)

        # advantage stream
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128 + metadata_dim, action_dim)

    def forward(self, state_embedding, state_metadata):
        # split concatenated embeddings: (batch, 3072) → 4 × (batch, 768)
        s, d, q, b = torch.split(state_embedding, 768, dim=-1)
        # 
        sq = torch.cat([s, q], dim=-1)
        dq = torch.cat([d, q], dim=-1)
        bq = torch.cat([b, q], dim=-1)

        # 
        sq = F.relu(self.fc_single(sq))
        dq = F.relu(self.fc_double(dq))
        bq = F.relu(self.fc_single(bq))

        #
        x = torch.cat([sq, dq, bq], dim=-1)

        # shared trunk
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

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

"""