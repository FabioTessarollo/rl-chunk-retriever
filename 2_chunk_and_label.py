import json
import re
import os

def chunk_text(text, chunk_size=100):
    # Split text into words and keep track of char spans
    words = []
    for match in re.finditer(r"\S+", text):
        words.append((match.group(), match.start(), match.end()))
    # Build chunks
    chunks = []
    total_words = len(words)
    for i in range(0, total_words, chunk_size):
        chunk_words = words[i:i+chunk_size]
        if not chunk_words:
            continue
        start_char = chunk_words[0][1]
        end_char = chunk_words[-1][2]
        chunk_text = text[start_char:end_char]
        chunks.append({'chunk_id': i // chunk_size, 'text': chunk_text})
    return chunks

def process_pages(pages_path, pages_out_path, chunk_size=100):
    pages_text_map = {}
    pages_chunks_map = {}
    with open(pages_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            entry = json.loads(line)
            page_id = entry['page_id']
            full_text = entry['full_page_text']
            chunks = chunk_text(full_text, chunk_size)
            pages_chunks_map[page_id] = chunks
            pages_text_map[page_id] = full_text

    pages_chunked = [
        {"page_id": page_id, "chunks": chunks}
        for page_id, chunks in pages_chunks_map.items()
    ]

    with open(pages_out_path, 'w', encoding='utf-8') as out:
        json.dump(pages_chunked, out, ensure_ascii=False, indent=2)

    return pages_text_map, pages_chunks_map

def process_relevant(relevant_path, relevant_out_path, pages_text_map, pages_chunks_map, chunk_size=100):
    relevant_output = []
    with open(relevant_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            entry = json.loads(line)
            query = entry['query']
            page_id = entry['page_id']
            paragraphs = entry['relevant_paragraphs']

            page_text = pages_text_map.get(page_id)
            page_chunks = pages_chunks_map.get(page_id)
            if not page_text or not page_chunks:
                continue

            relevant_chunks = set()
            for para in paragraphs:
                snippet = para[:50]  # use first 50 characters to locate it
                start_index = page_text.find(snippet)
                if start_index == -1:
                    continue
                end_index = start_index + len(para)

                for chunk in page_chunks:
                    chunk_id = chunk['chunk_id']
                    chunk_text = chunk['text']
                    chunk_start = page_text.find(chunk_text)
                    chunk_end = chunk_start + len(chunk_text)
                    if chunk_start == -1:
                        continue
                    if not (end_index <= chunk_start or start_index >= chunk_end):
                        relevant_chunks.add(chunk_id)

            relevant_output.append({
                "query": query,
                "page_id": page_id,
                "relevant_chunks": sorted(relevant_chunks)
            })

    with open(relevant_out_path, 'w', encoding='utf-8') as out:
        json.dump(relevant_output, out, ensure_ascii=False, indent=2)

def main():
    set = 'test'
    pages_path = f'data_extract/pages_{set}.jsonl'
    relevant_path = f'data_extract/relevant_paragraphs_{set}.jsonl'
    pages_out_path = f'data_chunks/pages_chunked_{set}.json'
    relevant_out_path = f'data_chunks/relevant_chunks_{set}.json'
    chunk_size = 100

    os.makedirs('data_chunks', exist_ok=True)
    pages_text_map, pages_chunks_map = process_pages(pages_path, pages_out_path, chunk_size)
    process_relevant(relevant_path, relevant_out_path, pages_text_map, pages_chunks_map, chunk_size)

if __name__ == '__main__':
    main()
