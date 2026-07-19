import torch
from retrieval.environment import Topic


def _make_topic(n_chunks=10, relevant=None, ranked=None):
    """Create a Topic with synthetic data."""
    emb_dim = 768
    query_emb = torch.randn(emb_dim)
    page = {i: torch.randn(emb_dim) for i in range(n_chunks)}
    if relevant is None:
        relevant = [2, 5]
    if ranked is None:
        ranked = list(range(n_chunks))
    return Topic(query_emb, page, ranked, relevant, max_exp_loops=1)


def test_initial_state_shape():
    topic = _make_topic()
    state_emb, state_meta, reward, done = topic.get_initial_step()
    assert state_emb.shape == (768 * 5,)  # 5 groups: curr, next, prev, query, bag
    assert state_meta.shape == (6,)


def test_skip_action_advances():
    topic = _make_topic()
    topic.get_initial_step()
    old_rank = topic.current_rank_chunk
    topic.step(0)  # skip
    assert topic.current_rank_chunk > old_rank


def test_take_single_adds_to_bag():
    topic = _make_topic()
    topic.get_initial_step()
    topic.step(1)  # take_single
    assert len(topic.bag_of_chunks) >= 1


def test_episode_terminates():
    topic = _make_topic(n_chunks=3, ranked=[0, 1, 2])
    topic.get_initial_step()
    for _ in range(10):
        _, _, _, done, truncated = topic.step(0)  # skip through all
        if done or truncated:
            break
    assert done or truncated


def test_reward_true_positive():
    topic = _make_topic(relevant=[0])
    topic.get_initial_step()
    # Taking chunk 0 which is relevant should give positive reward component
    _, _, reward, _, _ = topic.step(1)
    # TP reward = 1/3, mixed with F1 delta, should be positive
    assert reward > 0


def test_reward_false_positive():
    topic = _make_topic(relevant=[5], ranked=[3, 5, 7])
    topic.get_initial_step()
    # Chunk 3 is not relevant, taking it is FP
    _, _, reward, _, _ = topic.step(1)
    assert reward < 0


def test_f1_calculation():
    topic = _make_topic(relevant=[0, 1, 2])
    topic.get_initial_step()
    # Take chunk 0 (relevant)
    topic.step(1)
    # TP=1, FP=0, FN=2 → precision=1, recall=1/3, F1=0.5
    assert abs(topic.f1_score - 0.5) < 1e-6
