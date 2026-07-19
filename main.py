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

    # Get train data
    # extract('train', cfg)
    # chunk_and_label('train', cfg)
    # embed('train', cfg)

    # Get CosSim results and rankings
    # cos_sim(cfg)

    # Training of RL Model
    train(cfg)

    # Get test data
    # extract('test', cfg)
    # chunk_and_label('test', cfg)
    # embed('test', cfg)

    # Evaluate RL model on test data using trained model
    # test(cfg)

    # analyze(cfg)


if __name__ == '__main__':
    main()
