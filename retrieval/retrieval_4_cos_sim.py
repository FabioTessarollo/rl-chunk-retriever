import random
import json
import os
import numpy as np
import torch
from retrieval.Data import Data
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_sim(v1, v2):
    similarity = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return similarity.item()

def evaluate_with_threshold(data, query_ids, threshold, device):
    """Evaluate queries using a specific cosine similarity threshold"""
    recall_total = 0
    precision_total = 0
    f1_score_total = 0
    n = 0
    cosine_sim_results = []
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        query_desc = query.get("query_desc")
        page = data.get_page_chunks_dict(page_id)
        query_embedding = torch.tensor(query.get("query")).to(device)
        relevant_chunks = set(query.get("relevant_chunks"))
        
        # Get all chunks with similarity above threshold
        selected_chunks = set()
        for chunk_id, chunk_embedding in page.items():
            chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
            if chunk_similarity >= threshold:
                selected_chunks.add(chunk_id)
        
        # Count relevant retrieved
        num_relevant_retrieved = len(selected_chunks & relevant_chunks)
        num_relevant_total = len(relevant_chunks)
        num_retrieved = len(selected_chunks)
        
        n += 1
        
        # Compute recall and precision
        recall = num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0
        precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0
        
        recall_total += recall
        precision_total += precision
        
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_score_total += f1_score

        # Add query result to list
        cosine_sim_results.append({
            "query_desc": query_desc,
            "f1_score": f1_score,
            "cos_sim_retrieved_chunks": list(selected_chunks)
        })
    
    avg_recall = recall_total / n if n > 0 else 0
    avg_precision = precision_total / n if n > 0 else 0
    avg_f1_score = f1_score_total / n if n > 0 else 0
    
    return avg_recall, avg_precision, avg_f1_score, cosine_sim_results

def find_optimal_threshold(data, training_query_ids, device, threshold_range=(0.77, 0.83), step=0.01):
    """Find the optimal cosine similarity threshold using training data"""
    best_threshold = 0.0
    best_f1_score = 0.0
    
    print("Finding optimal threshold on training data...")
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    
    for threshold in thresholds:
        _, _, f1_score, _ = evaluate_with_threshold(data, training_query_ids, threshold, device)
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
        
        print(f"Threshold: {threshold:.3f}, F1: {f1_score:.4f}")
    
    print(f"\nBest threshold: {best_threshold:.3f} with F1: {best_f1_score:.4f}")
    return best_threshold

def get_rankings_with_threshold(data, query_ids, threshold, device, top_k, n_examples=2):
    """Get rankings for queries using the specified threshold"""
    rankings = {}
    example_query_ids = set(random.sample(query_ids, min(n_examples, len(query_ids))))
    
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        query_desc = query.get("query_desc")
        page = data.get_page_chunks_dict(page_id)
        query_embedding = torch.tensor(query.get("query")).to(device)
        relevant_chunks = set(query.get("relevant_chunks"))
        
        # Calculate similarities for single chunks
        chunks_similarity_dict = {}
        for chunk_id, chunk_embedding in page.items():
            chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
            if chunk_similarity >= threshold:
                chunks_similarity_dict[chunk_id] = chunk_similarity

        # sort by similarity
        top_chunks = dict(sorted(chunks_similarity_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])
        
        # Store the query info
        rankings[query_id] = {
            "query_desc": query_desc,
            "relevant_chunks": [chunk_id for chunk_id, _ in top_chunks.items()]
        }

        # Print example if query_id was selected
        if query_id in example_query_ids:
            selected_chunk_ids = {chunk_id for chunk_id, _ in top_chunks.items()}
            num_relevant_retrieved = len(selected_chunk_ids & relevant_chunks)
            
            print(f"\nExample for Query ID: {query_id}")
            print(f"Query Description: {query_desc}")
            print(f"Threshold: {threshold:.3f}")
            print(f"Relevant chunks: {sorted(list(relevant_chunks))}")
            print(f"Retrieved chunks: {[chunk_id for chunk_id, _ in top_chunks.items()]}")
            print(f"Relevant retrieved: {num_relevant_retrieved}/{len(relevant_chunks)}")
    
    return rankings

def cos_sim():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    pages_path = "data_3_embed/pages_chunked_emb_train.json"
    relevant_path = "data_3_embed/relevant_chunks_emb_train.json"
    cosine_sim_path = "data_4_cos_sim/cosine_sim_rank_threshold_only_single.json"

    data = Data(pages_path, relevant_path, cosine_sim_path)
    data.load_pages()
    data.load_relevant()
    data.load_cosine_sim()

    train_set, validation_set = data.balanced_split_query_ids(data.query_ids, 0.8)

    pages_path_test = f"data_3_embed/pages_chunked_emb_test.json"
    relevant_path_test = f"data_3_embed/relevant_chunks_emb_test.json"

    data_test = Data(pages_path_test, relevant_path_test)
    data_test.load_pages()
    data_test.load_relevant()

    n_examples = 2


    # Evaluate on training set with optimal threshold
    optimal_threshold = find_optimal_threshold(data, train_set, device)
    
    # Evaluate on validation set with optimal threshold
    print(f"\n=== Validation Set Results (Threshold: {optimal_threshold:.3f}) ===")
    val_recall, val_precision, val_f1, _ = evaluate_with_threshold(data, validation_set, optimal_threshold, device)
    print(f"Recall: {val_recall:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Find optimal threshold using full training data
    optimal_threshold = find_optimal_threshold(data, data.query_ids, device)

    # Evaluate on test set with optimal threshold
    print(f"\n=== Test Set Results (Threshold: {optimal_threshold:.3f}) ===")
    train_recall, train_precision, train_f1, cosine_sim_results = evaluate_with_threshold(data_test, data_test.query_ids, optimal_threshold, device)
    print(f"Recall: {train_recall:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"F1 Score: {train_f1:.4f}")
    
    # Get rankings for all queries using a threshold
    top_k = 40 # max rank size
    threshold = 0.77 # min similarity
    all_rankings = get_rankings_with_threshold(data, data.query_ids, threshold, device, top_k,n_examples)

    output_dir = "data_4_cos_sim"
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON files
    output_file = os.path.join(output_dir, f"cosine_sim_rank_threshold_only_single_{set}.json")
    with open(output_file, 'w') as f:
        json.dump(all_rankings, f, indent=2)

    # Save to JSON files
    output_file = os.path.join(output_dir, f"cosine_sim_rank_retrieved_test_single.json")
    with open(output_file, 'w') as f:
        json.dump(cosine_sim_results, f, indent=2)


