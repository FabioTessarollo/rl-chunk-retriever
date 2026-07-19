# Phase E — Reproducibility, Dev Quality & Final Checks

## Current State (post Phase A–D)

What's done:
- ✅ Config system (`config/config.yaml` + `settings.py` + `get_config()`)
- ✅ Device auto-detection and centralized seed management
- ✅ `evaluate.py` extracted from train (unified eval function with `EvalResult` dataclass)
- ✅ `plotting.py` extracted (train/val metrics + action counts)
- ✅ `feature_importance.py` extracted from test (ablation-based analysis)
- ✅ Dead code removed, commented blocks cleaned up
- ✅ Reward constants externalized to config (`reward_cfg`)

What remains (this phase):
- 🔲 Replace `print()` with `logging` across the entire codebase (~70 print calls)
- 🔲 Add `__init__.py` to `etl/` and `retrieval/`
- 🔲 CLI for `main.py` (run individual stages without editing code)
- 🔲 README with setup + usage docs
- 🔲 File renaming for consistency
- 🔲 Type hints on core classes
- 🔲 Tests

---

## Task E1 — Replace `print()` with `logging`

**Scope:** ~70 print calls across 9 files.

**Approach:**
1. Create a logging setup helper in `config/settings.py` (or a new `config/logging.py`) that configures:
   - Console handler (INFO level, concise format)
   - File handler (DEBUG level, timestamped, writes to `logs/` directory)
   - Suppress noisy third-party loggers (transformers, matplotlib)
2. Each module gets its own logger: `logger = logging.getLogger(__name__)`
3. Replace print calls with appropriate levels:
   - Progress/status → `logger.info()`
   - Results/metrics → `logger.info()`
   - Debug details (per-chunk actions) → `logger.debug()`
   - Warnings (missing data, fallbacks) → `logger.warning()`

**Files to update:**
- `etl/etl_1_extract.py` (~7 prints)
- `etl/etl_3_embed.py` (~1 print)
- `retrieval/Data.py` (~2 prints)
- `retrieval/retrieval_4_cos_sim.py` (~15 prints)
- `retrieval/retrieval_5_1_rl_train.py` (~6 prints)
- `retrieval/retrieval_5_2_rl_test.py` (~2 prints)
- `retrieval/retrieval_5_3_analysis.py` (~25 prints)
- `retrieval/feature_importance.py` (~5 prints)

**Note:** `retrieval_5_1_rl_train.py` already has its own `logging.basicConfig()` setup — unify it with the centralized logger.

---

## Task E2 — CLI for `main.py`

**Scope:** Replace the current hardcoded `train()` call with a proper CLI.

**Approach:** Use `argparse` (no extra dependency).

```python
# Target interface:
python main.py extract --dataset train
python main.py chunk --dataset train
python main.py embed --dataset train
python main.py cos-sim
python main.py train
python main.py test
python main.py analyze
python main.py pipeline          # run all stages
python main.py pipeline --from embed  # resume from a stage
```

**Implementation:**
1. Add `argparse` with subcommands for each stage
2. Add `--dataset` flag (train/test) for ETL stages
3. Add `--config` flag to override config path
4. Add `--log-level` flag (DEBUG/INFO/WARNING)
5. Wire each subcommand to its function, passing `cfg`

---

## Task E3 — File & Module Renaming

**Scope:** Rename files to consistent `snake_case`, drop numeric prefixes from retrieval.

| Current | New |
|---------|-----|
| `retrieval/Data.py` | `retrieval/data_loader.py` |
| `retrieval/DuelingDQN.py` | `retrieval/dueling_dqn.py` |
| `retrieval/Topic.py` | `retrieval/environment.py` |
| `retrieval/ReplayBuffer.py` | `retrieval/replay_buffer.py` |
| `retrieval/EarlyStopping.py` | `retrieval/early_stopping.py` |
| `retrieval/retrieval_4_cos_sim.py` | `retrieval/cosine_similarity.py` |
| `retrieval/retrieval_5_1_rl_train.py` | `retrieval/train.py` |
| `retrieval/retrieval_5_2_rl_test.py` | `retrieval/test.py` |
| `retrieval/retrieval_5_3_analysis.py` | `retrieval/analysis.py` |
| `etl/etl_1_extract.py` | `etl/extract.py` |
| `etl/etl_2_chunk_and_label.py` | `etl/chunk_and_label.py` |
| `etl/etl_3_embed.py` | `etl/embed.py` |

**After renaming, update:**
- All `from retrieval.X import Y` / `from etl.X import Y` across codebase
- `main.py` imports
- Add `__init__.py` to `etl/` and `retrieval/` with public API exports

---

## Task E4 — README

**Sections:**
1. **Project Overview** — What the system does (RL-based chunk retrieval for TREC CAR)
2. **Architecture** — ASCII/mermaid diagram of the pipeline stages
3. **Setup** — Python version, `uv sync`, config explanation
4. **Usage** — CLI examples for each stage
5. **Configuration** — Document `config.yaml` fields
6. **Project Structure** — Directory tree with descriptions
7. **Results** — Summary of best metrics (F1, recall, precision)

---

## Task E5 — Type Hints on Core Classes

**Scope:** Add type annotations to method signatures in core classes only. Don't annotate every local variable — just the public API.

**Files:**
- `retrieval/Data.py` — `load_pages()`, `load_relevant()`, `get_query_obj_from_id()`, etc.
- `retrieval/Topic.py` — `__init__()`, `get_initial_step()`, `step()`, `get_state_metadata()`
- `retrieval/DuelingDQN.py` — `__init__()`, `forward()`
- `retrieval/ReplayBuffer.py` — `add()`, `sample()`, `update_priorities()`
- `retrieval/EarlyStopping.py` — `__init__()`, `step()`
- `config/settings.py` — already typed ✅

---

## Task E6 — Tests

Unit tests for the components that have deterministic, isolated behavior. Use `pytest`.

### E6.1 — Config tests (`tests/test_config.py`)
```
test_get_config_loads_default        — loads config.yaml, returns Config object
test_config_attribute_access         — cfg.training.gamma == 0.99, cfg.device == "auto"
test_config_nested_access            — cfg.data.embed_dir == "data_3_embed"
test_get_device_auto_returns_device  — get_device() returns a torch.device
test_set_seed_deterministic          — after set_seed(), torch.rand() produces same value
```

### E6.2 — EarlyStopping tests (`tests/test_early_stopping.py`)
```
test_no_stop_on_improvement          — step() returns False when score improves
test_stop_after_patience_exhausted   — returns True after `patience` non-improving steps
test_reset_on_improvement            — counter resets when score beats best by delta_ratio
test_first_step_never_stops          — first call always returns False
test_exact_threshold_not_improvement — score == best * (1 + delta) is NOT counted as improvement
```

### E6.3 — ReplayBuffer tests (`tests/test_replay_buffer.py`)
```
test_add_and_sample                  — add N experiences, sample returns batch of correct size
test_capacity_overflow               — adding beyond capacity overwrites oldest
test_sum_tree_total                  — total priority matches sum of added priorities
test_priority_update                 — update() changes sampling distribution
test_simple_buffer_add_sample        — SimpleReplayBuffer basic add/sample
```

### E6.4 — DuelingDQN tests (`tests/test_dueling_dqn.py`)
```
test_output_shape                    — forward() returns tensor of shape (batch, action_dim)
test_return_streams                  — return_streams=True returns (q, value, advantage)
test_value_advantage_combine         — q = v + (a - a.mean)
test_gradient_flow                   — loss.backward() does not raise
```

### E6.5 — Topic (Environment) tests (`tests/test_environment.py`)
```
test_initial_state_shape             — get_initial_step() returns correct embedding + metadata dims
test_skip_action_advances            — action=0 moves to next ranked chunk
test_take_single_adds_to_bag         — action=1 adds current chunk to bag
test_episode_terminates              — done=True when ranked list exhausted
test_reward_true_positive            — taking a relevant chunk yields TP reward
test_reward_false_positive           — taking irrelevant chunk yields FP reward
test_f1_calculation                  — final F1 matches expected value for known TP/FP/FN counts
```

### E6.6 — Evaluate tests (`tests/test_evaluate.py`)
```
test_eval_result_defaults            — EvalResult() fields are zero/empty
test_evaluate_returns_eval_result    — returns EvalResult with populated metrics
```

### E6.7 — Integration smoke test (`tests/test_integration.py`)
```
test_train_one_epoch_no_crash        — train(cfg) with epochs=1 on tiny synthetic data runs without error
```

### Setup
1. Add `pytest` to `pyproject.toml` dev dependencies
2. Create `tests/` directory with `conftest.py` (shared fixtures: mini config, dummy data, device)
3. Create `tests/fixtures/` with minimal synthetic data (2 pages, 3 queries, few chunks)

---

## Task E7 — Final Checks

Run these after all E tasks are complete:

### Code quality
- [ ] `python -m py_compile main.py` — no syntax errors
- [ ] `python -c "from config import get_config; cfg = get_config(); print(cfg)"` — config loads
- [ ] `python main.py --help` — CLI shows all subcommands
- [ ] `grep -rn "print(" etl/ retrieval/ main.py` — no remaining raw prints (except `trec_car/`)
- [ ] `grep -rn '"mps"' retrieval/ etl/` — no hardcoded device strings

### Tests
- [ ] `pytest tests/ -v` — all tests pass
- [ ] `pytest tests/ --tb=short -q` — quick smoke check

### Imports & structure
- [ ] `python -c "import etl; import retrieval; import config"` — packages importable
- [ ] No circular imports (each module can be imported independently)

### Reproducibility
- [ ] `python main.py train` runs end-to-end on a fresh clone (after `uv sync` + data download)
- [ ] `python main.py test` loads the best model and produces results
- [ ] Changing `config.yaml` values (e.g. `seed`, `lr`) actually affects behavior

### Cleanup
- [ ] No `__pycache__/` committed to git
- [ ] `.gitignore` covers: `__pycache__/`, `*.pyc`, `data_*/`, `models/`, `.venv/`, `logs/`, `*.log`
- [ ] Remove `refactoringAB.md`, `refactoringCD.md`, `refactoringE.md` before final release (or keep in a `docs/` directory)

---

## Execution Order

```
E1 (logging) → E2 (CLI) → E3 (rename files) → E4 (README) → E5 (type hints) → E6 (tests) → E7 (final checks)
```

E1–E3 have cascading import effects, so do them in order. E4–E5 are independent. E6 should be last before E7, so tests validate the final structure.
