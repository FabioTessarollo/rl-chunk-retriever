import random
from collections import deque

import numpy as np


class SumTree:
    """
    Sum Tree data structure for efficient priority sampling.
    Each leaf stores a priority value and the internal nodes store the sum.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority changes up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority sum"""
        return self.tree[0]

    def add(self, p, data):
        """Add new experience with priority p"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update priority of experience at idx"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get experience with priority sum s"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, eps=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used
        self.beta = beta    # Importance sampling correction factor
        self.beta_increment = beta_increment
        self.eps = eps      # Small amount to avoid zero priority
        self.max_priority = 1.0

    def push(self, state_emb, state_meta, action, reward, next_emb, next_meta, done):
        """Add experience to buffer with maximum priority"""
        data = (state_emb, state_meta, action, reward, next_emb, next_meta, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size):
        """Sample batch_size experiences based on priority"""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (np.abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_emb, state_meta, action, reward, next_emb, next_meta, done):
        self.buffer.append((state_emb, state_meta, action, reward, next_emb, next_meta, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
