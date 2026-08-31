# RL Chunks Retriever

In Retrieval-Augmented Generation (RAG) systems, documents are typically divided into fixed-size chunks, which ignore the semantic structure of text. This causes fragmentation of information, complicating the retrieval of chunks that capture only a small portion of the relevant passage.
Strategies such as hierarchical, overlapping or semantic chunking are the prevailing approaches, but they may introduce noise, increase the number of tokens to process, or significantly raise computational costs.

As an alternative, we propose a Deep Reinforcement Learning model that sequentially navigates a similarity-based rank of fixed-size chunks and dynamically explores relationships between adjacent chunks. The model operates on top of pre-trained, frozen text embeddings, whose semantic representations are processed to enable a refined selection of chunks, improving upon a baseline solely relying on similarity scores.

To evaluate the proposed retrieval strategy, we construct a new benchmark derived from the TREC-CAR dataset and compare our model against a cosine similarity threshold baseline using the same frozen embeddings. Experimental results show a 12.8% improvement in F1 score and a 11\% increase in recall, without degrading precision.

These results point toward reinforcement learning as a viable enhancement to the standard retrieval stage in RAG pipelines, requiring no additional text to be embedded beyond the fixed-size chunks, and no Large Language Model (LLM) to be fine-tuned.

## Architecture

```
Raw TREC CAR data (CBOR)
  │  extract
  ▼
Pages + Relevant Paragraphs (JSONL)
  │  chunk
  ▼
50-word Chunks + Relevance Labels (JSON)
  │  embed
  ▼
Sentence Embeddings — e5-base-v2, 768-dim (JSON)
  │  cos-sim
  ▼
Cosine Similarity Rankings + Threshold Optimization
  │  train
  ▼
Dueling DQN Agent (5 actions: skip, take 1/2/3 chunks)
  │  test
  ▼
Test Set Predictions + Feature Importance
  │  analyze
  ▼
Merged Results, Metrics Comparison (RL vs Cosine Sim)
```

## Setup

**Requirements:** Python >= 3.11.9, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd rl-chunks-retriever
uv sync
```

Download the TREC CAR Y1 benchmark data into `data_0_raw/benchmarkY1-train/` and `data_0_raw/benchmarkY1-test/`.

## Usage

All pipeline stages are run through `main.py`:

```bash
# Run individual stages
uv run python main.py extract --dataset train
uv run python main.py chunk --dataset train
uv run python main.py embed --dataset train
uv run python main.py cos-sim
uv run python main.py train
uv run python main.py test
uv run python main.py analyze

# Run the full pipeline
uv run python main.py pipeline

# Resume from a specific stage
uv run python main.py pipeline --from train

# Options
uv run python main.py --log-level DEBUG --log-file logs/run.log train
uv run python main.py --config path/to/custom.yaml train
```

## Configuration

All settings are in [`config/config.yaml`](config/config.yaml):

| Section | Key fields |
|---------|-----------|
| `device` | `auto` / `mps` / `cuda` / `cpu` |
| `data.*` | Directory paths for each pipeline stage |
| `etl` | `chunk_size` (words per chunk), `embedding_model` |
| `retrieval.cos_sim` | `threshold`, `top_k`, `threshold_range` |
| `retrieval.reward` | TP/FP/FN reward values for the RL agent |
| `training` | All DQN hyperparameters (lr, gamma, epsilon, batch_size, etc.) |
| `model` | `embedding_dim`, `path` to saved checkpoint |

## Project Structure

```
main.py                      # CLI entry point
config/
  config.yaml                # All configuration
  settings.py                # Config loader, device detection, seed, logging
etl/
  extract.py                 # TREC CAR CBOR → JSONL pages + paragraphs
  chunk_and_label.py          # Text chunking + relevance labeling
  embed.py                   # Sentence embedding (e5-base-v2)
retrieval/
  cosine_similarity.py        # Threshold optimization + cosine sim baseline
  train.py                   # Dueling DQN training loop
  test.py                    # Model evaluation + feature importance
  analyze.py                 # Merge results, compute comparison metrics
  environment.py             # RL environment (state/action/reward)
  dueling_dqn.py             # Dueling DQN network architecture
  data_loader.py             # Data loading and query splitting
  replay_buffer.py           # Prioritized + simple replay buffers
  early_stopping.py          # Early stopping monitor
  evaluate.py                # Unified greedy evaluation
  feature_importance.py      # Ablation + gradient feature importance
  plotting.py                # Training visualization
trec_car/                    # TREC CAR data format parsers
```