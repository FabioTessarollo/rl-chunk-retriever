from retrieval.early_stopping import EarlyStopping


def test_first_step_never_stops():
    es = EarlyStopping(patience=3, delta_ratio=0.01)
    assert es.step(1.0) is False


def test_no_stop_on_improvement():
    es = EarlyStopping(patience=3, delta_ratio=0.01)
    es.step(1.0)
    assert es.step(1.02) is False
    assert es.step(1.04) is False


def test_stop_after_patience_exhausted():
    es = EarlyStopping(patience=3, delta_ratio=0.01)
    es.step(1.0)
    es.step(0.9)
    es.step(0.9)
    assert es.step(0.9) is True


def test_reset_on_improvement():
    es = EarlyStopping(patience=3, delta_ratio=0.01)
    es.step(1.0)
    es.step(0.9)  # counter=1
    es.step(0.9)  # counter=2
    es.step(1.02)  # improvement, counter resets
    assert es.counter == 0
    assert es.early_stop is False


def test_exact_threshold_not_improvement():
    es = EarlyStopping(patience=2, delta_ratio=0.01)
    es.step(1.0)
    # score < best * (1 + delta) = 1.01 → NOT improvement
    es.step(1.005)  # counter=1
    assert es.step(1.005) is True  # counter=2 >= patience
