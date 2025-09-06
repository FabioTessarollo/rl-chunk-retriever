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

def evaluate_with_threshold(data, query_ids, threshold, device):
    """Evaluate queries using a specific cosine similarity threshold"""
    recall_total = 0
    precision_total = 0
    f1_score_total = 0
    n = 0
    
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        page, _, _ = data.get_page_chunks_dict(page_id)
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
    
    avg_recall = recall_total / n if n > 0 else 0
    avg_precision = precision_total / n if n > 0 else 0
    avg_f1_score = f1_score_total / n if n > 0 else 0
    
    return avg_recall, avg_precision, avg_f1_score

def find_optimal_threshold(data, training_query_ids, device, threshold_range=(0.75, 0.85), step=0.01):
    """Find the optimal cosine similarity threshold using training data"""
    best_threshold = 0.0
    best_f1_score = 0.0
    
    print("Finding optimal threshold on training data...")
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    
    for threshold in thresholds:
        _, _, f1_score = evaluate_with_threshold(data, training_query_ids, threshold, device)
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
        
        print(f"Threshold: {threshold:.3f}, F1: {f1_score:.4f}")
    
    print(f"\nBest threshold: {best_threshold:.3f} with F1: {best_f1_score:.4f}")
    return best_threshold

def get_rankings_with_threshold(data, query_ids, threshold, device, n_examples=2):
    """Get rankings for queries using the specified threshold"""
    rankings = {}
    example_query_ids = set(random.sample(query_ids, min(n_examples, len(query_ids))))
    
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        query_desc = query.get("query_desc")
        page, _, _ = data.get_page_chunks_dict(page_id)
        query_embedding = torch.tensor(query.get("query")).to(device)
        relevant_chunks = set(query.get("relevant_chunks"))
        
        # Calculate similarities and filter by threshold
        chunks_similarity_dict = {}
        for chunk_id, chunk_embedding in page.items():
            chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
            if chunk_similarity >= threshold:
                chunks_similarity_dict[chunk_id] = chunk_similarity
        
        # Sort by similarity
        top_chunks = sorted(chunks_similarity_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Store the query info
        rankings[query_id] = {
            "query_desc": query_desc,
            "threshold": threshold,
            "relevant_chunks": [chunk_id for chunk_id, _ in top_chunks]
        }
        
        # Print example if query_id was selected
        if query_id in example_query_ids:
            selected_chunk_ids = {chunk_id for chunk_id, _ in top_chunks}
            num_relevant_retrieved = len(selected_chunk_ids & relevant_chunks)
            
            print(f"\nExample for Query ID: {query_id}")
            print(f"Query Description: {query_desc}")
            print(f"Threshold: {threshold:.3f}")
            print(f"Relevant chunks: {sorted(list(relevant_chunks))}")
            print(f"Retrieved chunks: {[chunk_id for chunk_id, _ in top_chunks]}")
            print(f"Relevant retrieved: {num_relevant_retrieved}/{len(relevant_chunks)}")
    
    return rankings

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pages_path = "data_chunks_emb/pages_chunked_emb.json"
    pages_doub_even_path = "data_chunks_emb/pages_doub_chunked_even.json"
    pages_doub_odd_path = "data_chunks_emb/pages_doub_chunked_odd.json"
    relevant_path = "data_chunks_emb/relevant_chunks_emb.json"
    n_examples = 2

    data = Data(pages_path, relevant_path, pages_doub_even_path, pages_doub_odd_path)
    data.load_pages()
    data.load_pages_even()
    data.load_pages_odd()
    data.load_relevant()

    # Split data into training and validation sets
    training_set, validation_set = data.split_query_ids(data.query_ids, 0.7)
    
    print(f"Training queries: {len(training_set)}")
    print(f"Validation queries: {len(validation_set)}")
    
    # Find optimal threshold using training data
    optimal_threshold = find_optimal_threshold(data, training_set, device)
    
    # Evaluate on training set with optimal threshold
    print(f"\n=== Training Set Results (Threshold: {optimal_threshold:.3f}) ===")
    train_recall, train_precision, train_f1 = evaluate_with_threshold(data, training_set, optimal_threshold, device)
    print(f"Recall: {train_recall:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"F1 Score: {train_f1:.4f}")
    
    # Evaluate on validation set with optimal threshold
    print(f"\n=== Validation Set Results (Threshold: {optimal_threshold:.3f}) ===")
    val_recall, val_precision, val_f1 = evaluate_with_threshold(data, validation_set, optimal_threshold, device)
    print(f"Recall: {val_recall:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Get rankings for all queries using optimal threshold
    print(f"\nGenerating rankings with optimal threshold...")
    all_rankings = get_rankings_with_threshold(data, data.query_ids, optimal_threshold, device, n_examples)
    
    # Create output directory if it doesn't exist
    output_dir = "data_chunks_cos_sim"
    os.makedirs(output_dir, exist_ok=True)

    # Save the rankings to JSON file
    output_file = os.path.join(output_dir, "cosine_sim_rank_threshold.json")
    with open(output_file, 'w') as f:
        json.dump(all_rankings, f, indent=2)

    # Save results summary
    results = {
        "optimal_threshold": optimal_threshold,
        "training_results": {
            "recall": train_recall,
            "precision": train_precision,
            "f1_score": train_f1,
            "num_queries": len(training_set)
        },
        "validation_results": {
            "recall": val_recall,
            "precision": val_precision,
            "f1_score": val_f1,
            "num_queries": len(validation_set)
        }
    }
    
    results_file = os.path.join(output_dir, "threshold_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRankings saved to: {output_file}")
    print(f"Results summary saved to: {results_file}")


if __name__ == "__main__":
    main()