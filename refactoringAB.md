# Refactoring Plan — Phase A & Phase B

## Overview

This plan covers two foundational refactoring phases for the RL Chunks Retriever project:

- **Phase A** — Configuration & Constants Extraction
- **Phase B** — Project Structure Reorganization

They can be done in either order, but doing A first means Phase B can already use config references in renamed files.

---

## Phase A — Configuration & Constants Extraction

### Goal
Centralize every hardcoded value (paths, hyperparameters, device, reward constants, thresholds) into a single config system so experiments can be changed without editing source code.

### A.1 — Create config infrastructure

Create `config/config.yaml` organized by pipeline stage, and a `config/settings.py` loader.

**Config YAML structure:**

```
device: auto               # auto | mps | cuda | cpu
seed: 1

data:
  raw_dir: data_0_raw
  extract_dir: data_1_extract
  chunk_dir: data_2_chunk_and_label
  embed_dir: data_3_embed
  cos_sim_dir: data_4_cos_sim
  analysis_dir: data_5_analysis
  models_dir: models

etl:
  chunk_size: 50
  embedding_model: intfloat/e5-base-v2

retrieval:
  cos_sim:
    threshold_range: [0.76, 0.86]
    threshold: 0.77
    top_k: 40
  reward:
    tp: 0.333
    fp: -0.167
    fn: -0.5

training:
  proj_dim: 512
  gamma: 0.99
  epsilon: 1.0
  epsilon_min: 0.1
  epsilon_decay: 0.99995
  batch_size: 32
  replay_capacity: 30000
  lr: 3.0e-5
  eta_min: 3.0e-6
  target_update: 4000
  epochs: 30
  max_exp_loops: 1
  action_dim: 5
  metadata_dim: 6
  dropout: 0.0
  per_alpha: 0.6
  per_beta: 0.4
  per_beta_increment: 0.001
  early_stopping_patience: 10
  early_stopping_delta: 0.001
  warm_up_epochs: 20
  neg_schedule_start: 0.30
  neg_schedule_end: 0.50
  neg_schedule_steps: 50
  weight_decay: 1.0e-4
  scheduler_t_max: 40
  train_split: 0.6

model:
  embedding_dim: 768
  path: models/rl-chunk-retriever_best.pt
```

**`config/settings.py`** — A simple loader using `pyyaml` (already a dependency) that reads the YAML into a nested dataclass or dict. Provide a `get_config(path="config/config.yaml")` function and a `get_device(cfg)` helper that resolves `"auto"` to the best available device.

### A.2 — Replace hardcoded paths

Every pipeline function currently hardcodes its input/output paths as string literals. Refactor each entry-point function to receive paths from config.

| File | Current hardcoded paths | Change |
|------|------------------------|--------|
| etl_1_extract.py → `extract()` | `data_0_raw/benchmarkY1-{set}/`, data_1_extract | Read from `cfg.data.raw_dir`, `cfg.data.extract_dir` |
| etl_2_chunk_and_label.py → `chunk_and_label()` | data_1_extract, data_2_chunk_and_label | Read from `cfg.data.extract_dir`, `cfg.data.chunk_dir` |
| etl_3_embed.py → `embed()` | data_2_chunk_and_label, data_3_embed, model name `intfloat/e5-base-v2` | Read from config |
| retrieval_4_cos_sim.py → `cos_sim()` | data_3_embed, data_4_cos_sim, threshold 0.77, top_k 40 | Read from config |
| retrieval_5_1_rl_train.py → `train()` | data_3_embed, data_4_cos_sim, all hyperparams, `"mps"` device | Read from config |
| retrieval_5_2_rl_test.py → `test()` | data_3_embed, model path rl-chunk-retriever_BEEEEEST22.pt, `"mps"` device | Read from config |
| retrieval_5_3_analysis.py → `analyze()` | data_5_analysis, data_4_cos_sim, data_2_chunk_and_label | Read from config |
| Data.py → `Data.__init__()` | `torch.device("mps")` hardcoded on line 15 | Accept device from caller (from config) |
| Topic.py → `Topic.__init__()` | Reward constants `TP=1/3, FP=-1/6, FN=-1/2` | Accept reward config dict or read from config |

### A.3 — Standardize entry-point function signatures (pattern consistency)

All ETL and retrieval entry-point functions should follow the same pattern:

```
def stage_name(cfg) -> None:
    # 1. Resolve input paths from cfg
    # 2. Resolve output paths from cfg
    # 3. Do work
    # 4. Write outputs
```

**Current inconsistencies:**
- `extract(set)` takes a string `"train"/"test"`
- `chunk_and_label(set, chunk_size=50)` takes set + a hardcoded default
- `embed(set)` takes a string
- `cos_sim()` takes nothing, hardcodes everything
- `train()` takes nothing, hardcodes everything
- `test()` takes nothing, hardcodes everything
- `analyze()` takes nothing, uses module-level constants

**Target pattern:** Each function receives `cfg` (the config object) and a `dataset: str` parameter where applicable (for ETL stages that process train/test independently). Internal helpers receive only the data they need, not the full config.

### A.4 — Device auto-detection

Replace all `torch.device("mps")` with a shared `get_device(cfg)` utility:
- Files affected: Data.py (line 15), retrieval_5_1_rl_train.py (line ~155), retrieval_5_2_rl_test.py (line ~395), retrieval_4_cos_sim.py (line ~192)
- `Data.__init__` should accept `device` as a parameter instead of hardcoding it

### A.5 — Centralize seed management

Replace scattered `random.seed(1)` / `torch.manual_seed(1)` (in retrieval_5_1_rl_train.py lines 22-23, retrieval_5_2_rl_test.py lines 19-20) with a single `set_seed(cfg.seed)` utility called once at startup.

---

## Phase B — Project Structure Reorganization

### Goal
Rename files to follow Python conventions, add proper packaging, and replace the comment-toggling in main.py with a CLI.

### B.1 — Rename retrieval module files to snake_case

| Current | New | Reason |
|---------|-----|--------|
| Data.py | `retrieval/data_loader.py` | PascalCase is for classes, not modules |
| DuelingDQN.py | `retrieval/dueling_dqn.py` | Same |
| Topic.py | `retrieval/environment.py` | Same + more descriptive name |
| ReplayBuffer.py | `retrieval/replay_buffer.py` | Same |
| EarlyStopping.py | `retrieval/early_stopping.py` | Same |
| retrieval_4_cos_sim.py | `retrieval/cosine_similarity.py` | Drop numeric prefix |
| retrieval_5_1_rl_train.py | `retrieval/train.py` | Drop numeric prefix |
| retrieval_5_2_rl_test.py | `retrieval/evaluate.py` | Drop numeric prefix, clarify purpose |
| retrieval_5_3_analysis.py | `retrieval/analysis.py` | Drop numeric prefix |

### B.2 — Update all imports

After renaming, update every `import` and `from ... import` across the codebase:
- main.py — all 7 imports
- `retrieval/train.py` — imports Data, Topic, DuelingDQN, PrioritizedReplayBuffer, EarlyStopping
- `retrieval/evaluate.py` — imports Data, Topic, DuelingDQN
- `retrieval/cosine_similarity.py` — imports Data

### B.3 — Add `__init__.py` files

Add to: etl, retrieval, `config/` — can be empty or re-export key symbols.

### B.4 — CLI for main.py

Replace the comment-toggling pattern with `argparse`:

```
python main.py extract --dataset train
python main.py chunk --dataset train --chunk-size 50
python main.py embed --dataset train
python main.py cos-sim
python main.py train
python main.py test
python main.py analyze
python main.py pipeline --dataset train   # runs extract → chunk → embed sequentially
```

Each subcommand loads config, optionally overrides `dataset`, and calls the corresponding function. This also eliminates the need for the `set` parameter — it comes from the CLI.

### B.5 — Clean up ETL file names (optional, lower priority)

| Current | New |
|---------|-----|
| etl_1_extract.py | `etl/extract.py` |
| etl_2_chunk_and_label.py | `etl/chunk_and_label.py` |
| etl_3_embed.py | `etl/embed.py` |

The numeric prefixes are unnecessary once there's a CLI that controls execution order.

### B.6 — Standardize ETL function pattern

All three ETL files already follow a similar pattern but with inconsistencies. Standardize to:

```
def stage_name(cfg, dataset: str) -> None:
    # Resolve I/O paths from cfg + dataset
    input_path = cfg.data.prev_stage_dir / f"file_{dataset}.json"
    output_path = cfg.data.this_stage_dir / f"file_{dataset}.json"
    os.makedirs(output_path.parent, exist_ok=True)

    # Load
    data = load(input_path)

    # Transform
    result = transform(data, **params_from_cfg)

    # Save
    save(result, output_path)
```

Current deviations:
- `extract()` builds paths inline from `base_dir` + fold logic, creates output dir, writes JSONL
- `chunk_and_label()` already close to this pattern — defines input/output at top, calls helpers
- `embed()` already close — defines paths at top, loads model, calls helpers

The retrieval entry points (`cos_sim`, `train`, `test`, `analyze`) have much more varied shapes and don't need to follow the ETL pattern exactly, but should all accept `cfg` as first parameter.

---

## Summary — Execution Order

| Step | Phase | Dependencies | Risk |
|------|-------|-------------|------|
| A.1 | Create `config/` infra | None | Low |
| A.2 | Replace hardcoded paths | A.1 | Medium (touches every file) |
| A.3 | Standardize function signatures | A.2 | Medium |
| A.4 | Device auto-detection | A.1 | Low |
| A.5 | Seed management | A.1 | Low |
| B.1 | Rename files | None (or after A) | Medium (breaks imports) |
| B.2 | Update imports | B.1 | Low (mechanical) |
| B.3 | Add `__init__.py` | B.1 | Low |
| B.4 | CLI for main.py | A.1 + B.2 | Medium |
| B.5 | Rename ETL files | B.1 | Low |
| B.6 | Standardize ETL pattern | A.3 | Low |

**Recommended order:** A.1 → A.4 → A.5 → A.2 → A.3 → B.1 → B.2 → B.3 → B.5 → B.6 → B.4

Steps A.4 and A.5 are small and can be done alongside A.1. B.4 (CLI) is best done last since it depends on stable function signatures.

---

## Out of scope (deferred)
- Splitting large files (Phase C)
- Dead code removal (Phase D)
- Vector DB / MLflow integration (Phase 2)
- Unit tests
- trec_car — third-party parser, left as-is

---

To get started, pick any step from the table above. A.1 (config infrastructure) is the natural first move — it's self-contained and everything else builds on it.

Created 2 todos
