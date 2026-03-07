from etl.etl_1_extract import extract
from etl.etl_2_chunk_and_label import chunk_and_label
from etl.etl_3_embed import embed
from retrieval.retrieval_4_cos_sim import cos_sim
from retrieval.retrieval_5_1_rl_train import train
from retrieval.retrieval_5_2_rl_test import test
from retrieval.retrieval_5_3_analysis import analyze



def main():

    # Get train data
    # extract('train')
    # chunk_and_label('train', chunk_size = 50)
    # embed('train')

    # Get CosSim results and rankings
    #cos_sim()

    # Traing of RL Model
    #train()

    # Get test data
    # extract('test')
    # chunk_and_label('test', chunk_size = 50)
    # embed('test')

    # Evaluate RL model on test data using trained model
    test()

    # analyze()



if __name__ == '__main__':
    main()