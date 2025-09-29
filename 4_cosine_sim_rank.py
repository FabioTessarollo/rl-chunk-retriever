import random
import json
import os
import numpy as np
import torch
from Data import Data
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_sim(v1, v2):
    similarity = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return similarity.item()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pages_path = "data_chunks_emb/pages_chunked_emb.json"
    pages_doub_even_path = "data_chunks_emb/pages_doub_chunked_even.json"
    pages_doub_odd_path = "data_chunks_emb/pages_doub_chunked_odd.json"
    relevant_path = "data_chunks_emb/relevant_chunks_emb.json"
    top_k = 30  # there are 3.5 relevant chunks per query on avg in fold 1
    n_examples = 2

    data = Data(pages_path, relevant_path, pages_doub_even_path, pages_doub_odd_path)
    data.load_pages()
    data.load_pages_even()
    data.load_pages_odd()
    data.load_relevant()

    recall_total = 0
    precision_total = 0
    f1_score_total = 0
    n = 0  # number of queries
    c = 0  # number of relevant chunks

    # Dictionary to store rankings for each query
    cosine_sim_rankings = {}

    # randomly select query ids to show as examples
    example_query_ids = set(random.sample(data.query_ids, min(n_examples, len(data.query_ids))))

    for query_id in data.query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        query_desc = query.get("query_desc")  # Get the query description
        page, _, _ = data.get_page_chunks_dict(page_id)
        query_embedding = torch.tensor(query.get("query")).to(device)
        relevant_chunks = set(query.get("relevant_chunks"))
        c += len(relevant_chunks)
        n += 1

        chunks_similarity_dict = {}
        for chunk_id, chunk_embedding in page.items():
            chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
            chunks_similarity_dict[chunk_id] = chunk_similarity

        avg_similarity = sum(chunks_similarity_dict.values()) / len(chunks_similarity_dict)

        # Sort by similarity, get top_k
        top_chunks = sorted(chunks_similarity_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_chunk_ids = {chunk_id for chunk_id, _ in top_chunks}

        avg_similarity_top10 = sum(similarity for _, similarity in top_chunks[:10]) / 10

        # Store the query info with description and ranked chunks
        cosine_sim_rankings[query_id] = {
            "query_desc": query_desc,
            "relevant_chunks": [chunk_id for chunk_id, _ in top_chunks],
            "avg_similarity": avg_similarity_top10
        }

        # Count relevant retrieved
        num_relevant_retrieved = len(top_chunk_ids & relevant_chunks)
        num_relevant_total = len(relevant_chunks)

        # Compute recall and precision
        if num_relevant_total > 0:
            recall = num_relevant_retrieved / num_relevant_total
            recall_total += recall

        precision = num_relevant_retrieved / top_k
        precision_total += precision

        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_score_total += f1_score

        # Print example if query_id was selected
        if query_id in example_query_ids:
            print(f"\nExample for Query ID: {query_id}")
            print(f"Query Description: {query_desc}")
            print(f"Relevant chunks: {sorted(list(relevant_chunks))}")
            print(f"Top retrieved chunks: {[chunk_id for chunk_id, _ in top_chunks[:top_k]]}")

    # Final average scores
    average_recall = recall_total / n if n > 0 else 0
    average_precision = precision_total / n if n > 0 else 0
    average_f1_score = f1_score_total / n if n > 0 else 0

    print(f"\nAverage Recall@{top_k}: {average_recall:.4f}")
    print(f"Average Precision@{top_k}: {average_precision:.4f}")
    print(f"Average F1@{top_k}: {average_f1_score:.4f}")

    # Create output directory if it doesn't exist
    output_dir = "data_chunks_cos_sim"
    os.makedirs(output_dir, exist_ok=True)

    # Save the rankings to JSON file
    output_file = os.path.join(output_dir, "cosine_sim_rank.json")
    with open(output_file, 'w') as f:
        json.dump(cosine_sim_rankings, f, indent=2)

    # print(f"\nCosine similarity rankings saved to: {output_file}")


if __name__ == "__main__":
    main()