# Refactoring Plan — Phase C (Separation of Concerns) & Phase D (Dead Code & Cleanup)

---

## Phase D — Dead Code & Cleanup

Phase D is the easier, lower-risk phase. Do it first to reduce noise before tackling Phase C.

### D1. Remove commented-out code blocks

**`main.py`** (lines 15–17, 20, 26–28, 31, 33)
- All ETL and test calls are commented out. These are now just noise since `main.py` already accepts stages via config/function calls. Remove the comments; the actual function calls + imports stay.

**`retrieval/retrieval_4_cos_sim.py`** (lines 248–265)
- 3 blocks of commented-out JSON save calls at the end of `cos_sim()`. The active save logic should remain; delete the commented duplicates.

**`retrieval/retrieval_5_2_rl_test.py`** (scattered)
- Commented-out action counters inside `evaluate()` (lines ~330-340: `#s += 1`, `#a1 += 1`, etc.)
- Variables `a2f`, `a2b`, `a3`, `a1`, `s` (line 300-305) are declared but never used because the counting is commented out. Remove both the variables and the comments.

**`retrieval/Topic.py`** (scattered)
- Commented-out lines: `#self.max_reward` (line 28), `#reward = reward/self.max_reward` in `take_single()`, `take_double()`, `take_prev_double()`. Remove all.
- Commented-out state metadata features in `get_state_metadata()` (lines ~63-66: `remaining_loops`, `prev_chunk_already_in_bag`, etc.). Remove.

**`retrieval/retrieval_5_1_rl_train.py`** (scattered)
- Commented-out `#action_log` and `#logging.info(...)` inside the training loop (~line 198-199). Remove.
- Commented-out model save at end of file (lines ~295-296: `# trained_model_path = ...`). Remove.

**`retrieval/DuelingDQN.py`** (lines 1-7)
- Duplicate import block: `torch`, `torch.nn`, `torch.nn.functional` are imported twice. Remove the duplicate (lines 1–3 or 5–7).

### D2. Delete artifact data files

**`data_4_cos_sim/`** — delete these files:
- `cosine_sim_rank_threshold_only_single_<class 'set'>.json` — bug artifact (a Python `set` type was string-interpolated into the filename)
- `cosine_sim_rank_threshold_only_single_test copy.json` — Finder copy artifact

Review and decide whether to keep:
- `cosine_sim_rank.json`, `cosine_sim_rank_threshold.json`, `cosine_sim_rank_threshold_test.json` — likely stale outputs from older pipeline runs. If not referenced by any code, delete.
- `double_similarities.json`, `single_similarities.json` — check if any code reads them; if not, delete.

### D3. Clean up model checkpoints

**`models/`** has 24 checkpoint files with inconsistent naming. Actions:
1. Identify which model file `cfg.model.path` points to — that's the "production" model. Keep it.
2. Delete `rl-chunk-retriever_BEEEEESTF1: 0.2803, Recall 0.6011, Precision 0.2257.pt` — the `:` and spaces in the filename cause cross-platform issues.
3. Delete `rl-chunk-retriever_BEEEEEST copy.pt` — Finder duplicate.
4. Delete `.DS_Store`.
5. Keep at most 2-3 checkpoints (best + latest). Move the rest out or delete after confirming with user.

### D4. Add `.gitignore`

No `.gitignore` exists. Create one with entries for:
```
__pycache__/
*.pyc
.DS_Store
models/
data_*/
*.log
*.png
feature_importance/
```

### D5. Remove unused variables

**`retrieval/retrieval_5_1_rl_train.py`**:
- `best_score` (line ~127) is initialized to 0, referenced only in the `extra_logger.info()` at the end but never updated. Either wire it up to actually track the best model, or remove it.
- `es` (EarlyStopping instance, line ~132) is created but never called (`.step()` is never invoked). Remove the instance and the import if early stopping isn't being used.

**`retrieval/retrieval_5_2_rl_test.py`**:
- `now_str` (line 17) is declared but never used. Remove.

---

## Phase C — Separation of Concerns

Phase C restructures large files that mix multiple responsibilities. Each task is independent — you can do them in any order.

### C1. Extract `evaluate()` from `retrieval_5_1_rl_train.py` into a shared module

**Problem:** Both `retrieval_5_1_rl_train.py` and `retrieval_5_2_rl_test.py` define their own `evaluate()` function with near-identical logic (iterate queries → run greedy policy → accumulate F1/recall/precision). The train version (~100 lines, lines 24-110) additionally tracks action counts and returns exploration probabilities. The test version (~80 lines, lines 293-380) additionally collects per-query results for JSON output.

**Action:**
1. Create `retrieval/evaluate.py` with a single `evaluate()` function that:
   - Takes the union of both signatures: `(data, query_ids, model, device, max_exp_loops, reward_cfg=None, track_actions=False, collect_results=False)`
   - Returns a result dataclass/namedtuple: `EvalResult(avg_reward, avg_f1, avg_recall, avg_precision, history=None, probs=None, results=None)`
   - The action-tracking (train version) and result-collection (test version) are toggled via the boolean flags
2. Update `retrieval_5_1_rl_train.py` to import from `retrieval/evaluate.py` — call with `track_actions=True`
3. Update `retrieval_5_2_rl_test.py` to import from `retrieval/evaluate.py` — call with `collect_results=True`
4. Delete both local `evaluate()` definitions

**Why:** Eliminates the most obvious code duplication in the repo. Currently, a bug fix in one `evaluate()` won't propagate to the other.

### C2. Extract plotting code from `retrieval_5_1_rl_train.py`

**Problem:** The last ~60 lines of `train()` (lines ~255-300) are four `matplotlib` plot blocks (F1, reward, recall, action counts). This is purely visualization, interleaved with training logic.

**Action:**
1. Create `retrieval/plotting.py` with functions:
   - `plot_train_val_metric(train_values, val_values, metric_name, filename)` — generic for F1/reward/recall
   - `plot_action_counts(history_df, filename)` — for the action distribution chart
2. Replace the inline plotting in `train()` with calls to these functions
3. The same plotting functions can later be reused by test/analysis scripts

### C3. Extract feature importance from `retrieval_5_2_rl_test.py`

**Problem:** `retrieval_5_2_rl_test.py` is ~430 lines. Of those, `compute_stream_ablation_importance()` is ~120 lines (lines 20-140) and `compute_stream_feature_importance()` is ~130 lines (lines 143-290). Together they're 250 lines of analysis code that has nothing to do with model evaluation. The `test()` function itself is only ~50 lines.

**Action:**
1. Create `retrieval/feature_importance.py` and move both functions there:
   - `compute_stream_ablation_importance(model, data, query_ids, device, max_exp_loops, n_samples)`
   - `compute_stream_feature_importance(model, data, query_ids, device, max_exp_loops, n_samples)`
2. Move their matplotlib/numpy imports with them
3. `retrieval_5_2_rl_test.py` becomes a clean ~50-line file: load model → evaluate → optionally call feature importance

### C4. Refactor action dispatch in `Topic.py`

**Problem:** Both `evaluate()` (in train and test) and the training loop repeat the same 5-way action dispatch:
```python
if action == 0:
    ... = topic.skip()
elif action == 1:
    ... = topic.take_single()
elif action == 2:
    ... = topic.take_double()
elif action == 3:
    ... = topic.take_prev_double()
elif action == 4:
    ... = topic.take_triple()
```
This pattern appears **4 times** across the codebase (train loop, train evaluate, test evaluate, feature importance). It also appears in the feature importance functions for stepping the environment.

**Action:**
Add a `step(action)` method to `Topic`:
```python
def step(self, action):
    dispatch = [self.skip, self.take_single, self.take_double, self.take_prev_double, self.take_triple]
    return dispatch[action]()
```
Then replace all 4+ occurrences with `topic.step(action)`.

### C5. Deduplicate reward logic in `Topic.py`

**Problem:** `Topic.py` is ~280 lines. The action methods (`skip`, `take_single`, `take_double`, `take_prev_double`, `take_triple`) each independently:
1. Compute a raw reward based on TP/FP/FN
2. Update bag embeddings with copy-paste logic
3. Call `_advance_rank()`
4. Get new state
5. Blend reward: `reward = (reward * 3 + self.reward_f1) / 4`

The bag-update logic (`if c not in bag: n = len(bag); embedding = (embedding * n + emb) / (n+1); bag.append(c)`) is duplicated 7 times across the methods.

**Action:**
1. Extract `_add_to_bag(self, chunk_id, embedding)` helper method for the repeated bag-update pattern
2. Extract `_compute_step_return(self, raw_reward)` that does the `_advance_rank() → get_state → blend reward → return tuple` sequence
3. Each action method becomes: compute raw reward → call `_add_to_bag()` for relevant chunks → call `_compute_step_return()`
4. This should roughly halve the line count of the action methods

### C6. Separate analysis plotting from metrics in `retrieval_5_3_analysis.py`

**Problem:** `retrieval_5_3_analysis.py` (~350 lines) mixes:
- Data merging logic (`merge_json_data`, ~60 lines)
- Metric computation (`f1_score`, `calculate_metrics`, `process_all_queries`, ~150 lines)
- Plotting (`plot_relevance_coverage`, ~50 lines)
- Orchestration (`analyze()`, ~40 lines)

**Action:**
1. Move `plot_relevance_coverage()` into `retrieval/plotting.py` (from C2)
2. Consider moving `f1_score()` into a small `retrieval/metrics.py` util — it's a generic metric that could be reused elsewhere
3. Keep `merge_json_data`, `process_all_queries`, `calculate_metrics`, and `analyze()` in the analysis file

This is lower priority than C1-C5 since the analysis file is already somewhat organized.

---

## Execution Order (recommended)

```
D1 (commented code)  ← quick, no risk
D2 (artifact files)  ← quick, no risk  
D4 (.gitignore)      ← quick, no risk
D5 (unused vars)     ← quick, low risk
D3 (model cleanup)   ← needs user input on which to keep
──────────────────────
C4 (Topic.step())    ← small, high reuse
C5 (Topic dedup)     ← contained in one file  
C1 (evaluate dedup)  ← highest value
C3 (feature importance extraction) ← straightforward move
C2 (plotting)        ← straightforward move
C6 (analysis split)  ← lowest priority
```

## Files Modified/Created

| Task | Files Modified | Files Created |
|------|---------------|---------------|
| D1 | `main.py`, `retrieval/retrieval_4_cos_sim.py`, `retrieval/retrieval_5_2_rl_test.py`, `retrieval/Topic.py`, `retrieval/retrieval_5_1_rl_train.py`, `retrieval/DuelingDQN.py` | — |
| D2 | — | — (deletions only) |
| D3 | — | — (deletions only) |
| D4 | — | `.gitignore` |
| D5 | `retrieval/retrieval_5_1_rl_train.py`, `retrieval/retrieval_5_2_rl_test.py` | — |
| C1 | `retrieval/retrieval_5_1_rl_train.py`, `retrieval/retrieval_5_2_rl_test.py` | `retrieval/evaluate.py` |
| C2 | `retrieval/retrieval_5_1_rl_train.py` | `retrieval/plotting.py` |
| C3 | `retrieval/retrieval_5_2_rl_test.py` | `retrieval/feature_importance.py` |
| C4 | `retrieval/Topic.py`, `retrieval/retrieval_5_1_rl_train.py`, `retrieval/retrieval_5_2_rl_test.py` (+ feature_importance if C3 done) | — |
| C5 | `retrieval/Topic.py` | — |
| C6 | `retrieval/retrieval_5_3_analysis.py` | (add to `retrieval/plotting.py`, optionally `retrieval/metrics.py`) |

## Verification

After each task:
1. Run `python main.py` (with `train()` as the active call) — should execute without import errors
2. Run `python -c "from retrieval.retrieval_5_2_rl_test import test"` — verify imports resolve
3. Run `python -c "from retrieval.retrieval_5_3_analysis import analyze"` — verify imports resolve
4. If a training run was previously working, verify a short training run (1-2 epochs) still produces the same outputs
5. After C4: verify `topic.step(action)` returns identical tuples to the if/elif dispatch for all 5 actions
