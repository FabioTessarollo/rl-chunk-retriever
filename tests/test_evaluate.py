from retrieval.evaluate import EvalResult


def test_eval_result_defaults():
    r = EvalResult()
    assert r.avg_reward == 0
    assert r.avg_f1 == 0
    assert r.avg_recall == 0
    assert r.avg_precision == 0
    assert r.action_counts == {}
    assert r.probs == []
    assert r.results == []
