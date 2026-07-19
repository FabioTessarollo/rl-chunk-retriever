import json
import logging
import os
import torch

from retrieval.data_loader import Data
from retrieval.dueling_dqn import DuelingDQN
from retrieval.evaluate import evaluate
from retrieval.feature_importance import compute_stream_ablation_importance, compute_stream_feature_importance
from config import get_config, get_device, set_seed

logger = logging.getLogger(__name__)


def test(cfg=None):
    if cfg is None:
        cfg = get_config()

    set_seed(cfg)
    device = get_device(cfg)
    logger.info(f"Using device: {device}")

    embed_dir = cfg.data.embed_dir
    cos_sim_dir = cfg.data.cos_sim_dir
    t = cfg.training

    pages_path_test = f"{embed_dir}/pages_chunked_emb_test.json"
    relevant_path_test = f"{embed_dir}/relevant_chunks_emb_test.json"
    cosine_sim_path_test = f"{cos_sim_dir}/cosine_sim_rank_threshold_only_single_test.json"

    data_test = Data(pages_path_test, relevant_path_test, cosine_sim_path_test, device)
    data_test.load_pages()
    data_test.load_relevant()
    data_test.load_cosine_sim()

    model = DuelingDQN(metadata_dim=t.metadata_dim, action_dim=t.action_dim, proj_dim=t.proj_dim, dropout_p=t.dropout, embedding_dim=cfg.model.embedding_dim).to(device)
    model.load_state_dict(torch.load(cfg.model.path, map_location="cpu"))
    model.eval()

    result = evaluate(data_test, data_test.query_ids, model, device, t.max_exp_loops, collect_results=True)

    logger.info(f"Test - Reward: {result.avg_reward:.4f}, F1: {result.avg_f1:.4f}, "
          f"Recall: {result.avg_recall:.4f}, Precision: {result.avg_precision:.4f}")

    analysis_dir = cfg.data.analysis_dir
    os.makedirs(analysis_dir, exist_ok=True)
    with open(f'{analysis_dir}/rl_model_retrieved_test_single.json', 'w') as f:
        json.dump(result.results, f, indent=2)

    compute_stream_ablation_importance(
        model, data_test, data_test.query_ids, device, max_exp_loops=t.max_exp_loops, n_samples=200
    )