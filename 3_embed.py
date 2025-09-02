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

def embed_merged_chunks(input_path, output_path_even, output_path_odd, model):
    """
    Computes embeddings for merged chunks:
    - Even pairs: [0,1], [2,3], [4,5] ... saved with ids 0, 2, 4 ...
    - Odd pairs: [1,2], [3,4], [5,6] ... saved with ids 1, 3, 5 ...
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    pages_even = []
    pages_odd = []

    for page in tqdm(pages, desc='Processing merged chunks'):
        chunks = page.get('chunks', [])
        
        # Create new page structures for even and odd merged chunks
        page_even = {key: value for key, value in page.items() if key != 'chunks'}
        page_odd = {key: value for key, value in page.items() if key != 'chunks'}
        page_even['chunks'] = []
        page_odd['chunks'] = []

        # Process even pairs: [0,1], [2,3], [4,5] ...
        for i in range(0, len(chunks) - 1, 2):
            if i + 1 < len(chunks):
                # Get text from both chunks
                text1 = chunks[i].get('text', '')
                text2 = chunks[i + 1].get('text', '')
                merged_text = text1 + ' ' + text2
                
                # Create merged chunk with embedding
                emb = model.encode(merged_text, normalize_embeddings=True)
                merged_chunk = {
                    'chunk_id': i,  # Use first chunk id (even)
                    'embedding': emb.tolist(),
                    'original_chunks': [i, i + 1]
                }
                # Copy other metadata from first chunk (excluding text and embedding)
                for key, value in chunks[i].items():
                    if key not in ['text', 'embedding']:
                        merged_chunk[key] = value
                
                page_even['chunks'].append(merged_chunk)

        # Process odd pairs: [1,2], [3,4], [5,6] ...
        for i in range(1, len(chunks) - 1, 2):
            if i + 1 < len(chunks):
                # Get text from both chunks
                text1 = chunks[i].get('text', '')
                text2 = chunks[i + 1].get('text', '')
                merged_text = text1 + ' ' + text2
                
                # Create merged chunk with embedding
                emb = model.encode(merged_text, normalize_embeddings=True)
                merged_chunk = {
                    'chunk_id': i,  # Use first chunk id (odd)
                    'embedding': emb.tolist(),
                    'original_chunks': [i, i + 1]
                }
                # Copy other metadata from first chunk (excluding text and embedding)
                for key, value in chunks[i].items():
                    if key not in ['text', 'embedding']:
                        merged_chunk[key] = value
                
                page_odd['chunks'].append(merged_chunk)

        pages_even.append(page_even)
        pages_odd.append(page_odd)

    # Save even merged chunks
    os.makedirs(os.path.dirname(output_path_even), exist_ok=True)
    with open(output_path_even, 'w', encoding='utf-8') as f:
        json.dump(pages_even, f)

    # Save odd merged chunks
    os.makedirs(os.path.dirname(output_path_odd), exist_ok=True)
    with open(output_path_odd, 'w', encoding='utf-8') as f:
        json.dump(pages_odd, f)


def main():
    model_name = 'intfloat/e5-base-v2'
    print(f'Loading model {model_name}...')
    model = SentenceTransformer(model_name) # this is already doing L2 Norm

    pages_in = 'data_chunks/pages_chunked.json'
    pages_out = 'data_chunks_emb/pages_chunked_emb.json'
    rels_in = 'data_chunks/relevant_chunks.json'
    rels_out = 'data_chunks_emb/relevant_chunks_emb.json'
    
    # New output paths for merged chunks
    pages_doub_even_out = 'data_chunks_emb/pages_doub_chunked_even.json'
    pages_doub_odd_out = 'data_chunks_emb/pages_doub_chunked_odd.json'

    embed_pages(pages_in, pages_out, model)
    embed_relevant(rels_in, rels_out, model)
    embed_merged_chunks(pages_in, pages_doub_even_out, pages_doub_odd_out, model)

    print('Embedding complete.')

if __name__ == '__main__':
    main()