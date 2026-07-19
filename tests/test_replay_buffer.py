import numpy as np
from retrieval.replay_buffer import PrioritizedReplayBuffer, SumTree, SimpleReplayBuffer
import torch


def _make_experience():
    return (torch.zeros(5), torch.zeros(3), 0, 1.0, torch.zeros(5), torch.zeros(3), False)


def test_add_and_sample():
    buf = PrioritizedReplayBuffer(capacity=100)
    for _ in range(20):
        s_e, s_m, a, r, n_e, n_m, d = _make_experience()
        buf.push(s_e, s_m, a, r, n_e, n_m, d)
    batch, idxs, weights = buf.sample(8)
    assert len(batch) == 8
    assert len(idxs) == 8
    assert len(weights) == 8


def test_capacity_overflow():
    buf = PrioritizedReplayBuffer(capacity=4)
    for i in range(10):
        buf.push(torch.zeros(5), torch.zeros(3), i, 1.0, torch.zeros(5), torch.zeros(3), False)
    assert len(buf) == 4


def test_sum_tree_total():
    tree = SumTree(8)
    tree.add(1.0, "a")
    tree.add(2.0, "b")
    tree.add(3.0, "c")
    assert np.isclose(tree.total(), 6.0)


def test_priority_update():
    tree = SumTree(8)
    tree.add(1.0, "a")
    old_total = tree.total()
    idx = tree.capacity - 1  # first leaf
    tree.update(idx, 5.0)
    assert tree.total() > old_total


def test_simple_buffer_add_sample():
    buf = SimpleReplayBuffer(capacity=100)
    for _ in range(20):
        s_e, s_m, a, r, n_e, n_m, d = _make_experience()
        buf.push(s_e, s_m, a, r, n_e, n_m, d)
    batch = buf.sample(5)
    assert len(batch) == 5
