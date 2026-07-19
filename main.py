from etl.etl_1_extract import extract
from etl.etl_2_chunk_and_label import chunk_and_label
from etl.etl_3_embed import embed
from retrieval.retrieval_4_cos_sim import cos_sim
from retrieval.retrieval_5_1_rl_train import train
from retrieval.retrieval_5_2_rl_test import test
from retrieval.retrieval_5_3_analysis import analyze
from config import get_config


def main():
    cfg = get_config()

    train(cfg)


if __name__ == '__main__':
    main()
