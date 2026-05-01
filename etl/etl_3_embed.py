import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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


def embed(set):

    # input
    pages_in = f'data_2_chunk_and_label/pages_chunked_{set}.json'
    rels_in = f'data_2_chunk_and_label/relevant_chunks_{set}.json'

    # output
    pages_out = f'data_3_embed/pages_chunked_emb_{set}.json'
    rels_out = f'data_3_embed/relevant_chunks_emb_{set}.json'

    model_name = 'intfloat/e5-base-v2'
    print(f'Loading model {model_name}...')
    model = SentenceTransformer(model_name) # this is already doing L2 Norm

    embed_pages(pages_in, pages_out, model)
    embed_relevant(rels_in, rels_out, model)