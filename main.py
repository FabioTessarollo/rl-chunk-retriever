from etl.etl_1_extract import extract
from etl.etl_2_chunk_and_label import chunk_and_label
from etl.etl_3_embed import embed
from retrieval.retrieval_4_cos_sim import cos_sim
from retrieval.retrieval_5_rl_train import train
from retrieval.retrieval_6_rl_test import *


def main():
    set = 'test'
    pages_output, queries_output = extract(set)
    chunk_and_label(set, pages_output, queries_output)



if __name__ == '__main__':
    main()