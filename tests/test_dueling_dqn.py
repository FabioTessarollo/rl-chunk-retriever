import torch
from retrieval.dueling_dqn import DuelingDQN


def test_output_shape():
    model = DuelingDQN(metadata_dim=6, action_dim=5, proj_dim=64, embedding_dim=32)
    emb = torch.randn(4, 32 * 5)  # batch=4, 5 groups of embeddings
    meta = torch.randn(4, 6)
    q = model(emb, meta)
    assert q.shape == (4, 5)


def test_return_streams():
    model = DuelingDQN(metadata_dim=6, action_dim=5, proj_dim=64, embedding_dim=32)
    emb = torch.randn(2, 32 * 5)
    meta = torch.randn(2, 6)
    q, v, a = model(emb, meta, return_streams=True)
    assert q.shape == (2, 5)
    assert v.shape == (2, 1)
    assert a.shape == (2, 5)


def test_value_advantage_combine():
    model = DuelingDQN(metadata_dim=6, action_dim=5, proj_dim=64, embedding_dim=32)
    emb = torch.randn(1, 32 * 5)
    meta = torch.randn(1, 6)
    q, v, a = model(emb, meta, return_streams=True)
    expected = v + a - a.mean(dim=-1, keepdim=True)
    assert torch.allclose(q, expected, atol=1e-6)


def test_gradient_flow():
    model = DuelingDQN(metadata_dim=6, action_dim=5, proj_dim=64, embedding_dim=32)
    emb = torch.randn(2, 32 * 5)
    meta = torch.randn(2, 6)
    q = model(emb, meta)
    loss = q.sum()
    loss.backward()
    # Check that at least one parameter has gradients
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
