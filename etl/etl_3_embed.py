import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import get_config

def embed_pages(input_path, output_path, model):
    with open(input_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    for page in tqdm(pages, desc='Embedding pages'):
        for chunk in page.get('chunks', []):
            text = chunk.pop('text', None)
            if text is not None:
                emb = model.encode(text, normalize_embeddings=True)
                chunk['embedding'] = emb.tolist()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f)

def embed_relevant(input_path, output_path, model):
    with open(input_path, 'r', encoding='utf-8') as f:
        rels = json.load(f)

    query_counter = 0
    for item in tqdm(rels, desc='Embedding queries'):
        query_text = item.pop('query', None)
        if query_text is not None:
            emb = model.encode(query_text, normalize_embeddings=True)
            item['query_id'] = query_counter
            item['query_desc'] = query_text
            query_counter += 1
            item['query'] = emb.tolist()
        # other fields (page_id, relevant_chunks) remain

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rels, f)


def embed(dataset, cfg=None):
    if cfg is None:
        cfg = get_config()

    chunk_dir = cfg.data.chunk_dir
    embed_dir = cfg.data.embed_dir

    # input
    pages_in = f'{chunk_dir}/pages_chunked_{dataset}.json'
    rels_in = f'{chunk_dir}/relevant_chunks_{dataset}.json'

    # output
    pages_out = f'{embed_dir}/pages_chunked_emb_{dataset}.json'
    rels_out = f'{embed_dir}/relevant_chunks_emb_{dataset}.json'

    model_name = cfg.etl.embedding_model
    print(f'Loading model {model_name}...')
    model = SentenceTransformer(model_name) # this is already doing L2 Norm

    embed_pages(pages_in, pages_out, model)
    embed_relevant(rels_in, rels_out, model)