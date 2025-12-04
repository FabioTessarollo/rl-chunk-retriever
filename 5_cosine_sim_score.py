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
    cosine_sim_results = []
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        query_desc = query.get("query_desc")
        page, page_even, page_odd = data.get_page_chunks_dict(page_id)
        query_embedding = torch.tensor(query.get("query")).to(device)
        relevant_chunks = set(query.get("relevant_chunks"))
        
        # Get all chunks with similarity above threshold
        selected_chunks = set()
        for chunk_id, chunk_embedding in page.items():
            chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
            if chunk_similarity >= threshold:
                selected_chunks.add(chunk_id)
        
        # Get all double even chunks with similarity above threshold
        # for chunk_id, chunk_embedding in page_even.items():
        #     chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
        #     if chunk_similarity >= threshold:
        #         selected_chunks.add(chunk_id)
        #         selected_chunks.add(chunk_id + 1)

        # Get all double odd chunks with similarity above threshold
        # for chunk_id, chunk_embedding in page_odd.items():
        #     chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
        #     if chunk_similarity >= threshold:
        #         selected_chunks.add(chunk_id)
        #         selected_chunks.add(chunk_id + 1)
        
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
    single_similarities = {}
    double_similarities = {}
    example_query_ids = set(random.sample(query_ids, min(n_examples, len(query_ids))))
    
    for query_id in query_ids:
        query = data.get_query_obj_from_id(query_id)
        page_id = query.get("page_id")
        query_desc = query.get("query_desc")
        page, page_even, page_odd = data.get_page_chunks_dict(page_id)
        query_embedding = torch.tensor(query.get("query")).to(device)
        relevant_chunks = set(query.get("relevant_chunks"))

        global_min = 1
        global_max = 0
        
        # Calculate similarities for single chunks
        chunks_similarity_dict = {}
        for chunk_id, chunk_embedding in page.items():
            chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
            if chunk_similarity >= threshold:
                chunks_similarity_dict[chunk_id] = chunk_similarity
                if chunk_similarity < global_min:
                    global_min = chunk_similarity
                if chunk_similarity > global_max:
                    global_max = chunk_similarity

        # Calculate similarities for double paired even chunks
        # chunks_similarity_dict_even_pairs = {}
        # for chunk_id, chunk_embedding in page_even.items():
        #     chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
        #     if chunk_similarity >= threshold:
        #         chunks_similarity_dict_even_pairs[chunk_id] = chunk_similarity
        #         if chunk_similarity < global_min:
        #             global_min = chunk_similarity
        #         if chunk_similarity > global_max:
        #             global_max = chunk_similarity

        # Calculate similarities for double paired odd chunks
        # chunks_similarity_dict_odd_pairs = {}
        # for chunk_id, chunk_embedding in page_odd.items():
        #     chunk_similarity = get_cosine_sim(chunk_embedding, query_embedding)
        #     if chunk_similarity >= threshold:
        #         chunks_similarity_dict_odd_pairs[chunk_id] = chunk_similarity
        #         if chunk_similarity < global_min:
        #             global_min = chunk_similarity
        #         if chunk_similarity > global_max:
        #             global_max = chunk_similarity

        range_similarity = global_max - global_min

        # merge base dict
        # merged = {
        #     k: max(
        #         chunks_similarity_dict.get(k, float('-inf'))#,
        #         # chunks_similarity_dict_even_pairs.get(k, float('-inf')),
        #         # chunks_similarity_dict_odd_pairs.get(k, float('-inf'))
        #     )
        #     for k in (
        #         set(chunks_similarity_dict)
        #         # | set(chunks_similarity_dict_even_pairs)
        #         # | set(chunks_similarity_dict_odd_pairs)
        #     )
        # }

        # sort by similarity
        top_chunks = dict(sorted(chunks_similarity_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])
        
        # Store the query info
        # rankings[query_id] = {
        #     "query_desc": query_desc,
        #     "relevant_chunks": [chunk_id for chunk_id, _ in top_chunks.items()]
        # }

        # single_similarities[query_id] = {
        #     "query_desc": query_desc,
        #     "similarities": {k: (v - global_min) / range_similarity for k, v in chunks_similarity_dict.items()}
        # }

        # double_similarities[query_id] = {
        #     "query_desc": query_desc,
        #     "similarities": {k: (v - global_min) / range_similarity for k, v in (chunks_similarity_dict_even_pairs | chunks_similarity_dict_odd_pairs).items()}
        # }
        double_similarities = None

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
    
    return rankings, single_similarities, double_similarities

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pages_path = f"data_chunks_emb/pages_chunked_emb_train.json"
    pages_doub_even_path = f"data_chunks_emb/pages_doub_chunked_even_train.json"
    pages_doub_odd_path = f"data_chunks_emb/pages_doub_chunked_odd_train.json"
    relevant_path = f"data_chunks_emb/relevant_chunks_emb_train.json"
    cosine_sim_path = f"data_chunks_cos_sim/cosine_sim_rank_threshold.json"

    pages_path_test = f"data_chunks_emb/pages_chunked_emb_test.json"
    pages_doub_even_path_test = f"data_chunks_emb/pages_doub_chunked_even_test.json"
    pages_doub_odd_path_test = f"data_chunks_emb/pages_doub_chunked_odd_test.json"
    relevant_path_test = f"data_chunks_emb/relevant_chunks_emb_test.json"

    data = Data(pages_path, relevant_path, pages_doub_even_path, pages_doub_odd_path, cosine_sim_path)
    data.load_pages()
    data.load_pages_even()
    data.load_pages_odd()
    data.load_relevant()
    data.load_cosine_sim()

    data_test = Data(pages_path_test, relevant_path_test, pages_doub_even_path_test, pages_doub_odd_path_test)
    data_test.load_pages()
    data_test.load_pages_even()
    data_test.load_pages_odd()
    data_test.load_relevant()

    n_examples = 2

    # Fair queries
    fair_query_ids, superdifficult_query_ids = data.get_query_ids_by_difficulty()

    # Split data into training and validation sets
    training_set, validation_set = data.balanced_split_query_ids(fair_query_ids, 1)
    
    print(f"Training queries: {len(training_set)}")
    print(f"Validation queries: {len(validation_set)}")
    
    # Find optimal threshold using training data
    optimal_threshold = find_optimal_threshold(data, training_set, device)
    
    # Evaluate on training set with optimal threshold
    print(f"\n=== Training Set Results (Threshold: {optimal_threshold:.3f}) ===")
    train_recall, train_precision, train_f1, _ = evaluate_with_threshold(data, training_set, optimal_threshold, device)
    print(f"Recall: {train_recall:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"F1 Score: {train_f1:.4f}")
    
    # Evaluate on validation set with optimal threshold
    # print(f"\n=== Validation Set Results (Threshold: {optimal_threshold:.3f}) ===")
    # val_recall, val_precision, val_f1 = evaluate_with_threshold(data, validation_set, optimal_threshold, device)
    # print(f"Recall: {val_recall:.4f}")
    # print(f"Precision: {val_precision:.4f}")
    # print(f"F1 Score: {val_f1:.4f}")

    # Evaluate on test set with optimal threshold
    print(f"\n=== Test Set Results (Threshold: {optimal_threshold:.3f}) ===")
    val_recall, val_precision, val_f1, cosine_sim_results  = evaluate_with_threshold(data_test, data_test.query_ids, optimal_threshold, device)
    print(f"Recall: {val_recall:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Get rankings for all queries using optimal threshold
    print(f"\nGenerating rankings with optimal threshold...")
    top_k = 40
    threshold = 0.77
    all_rankings, single_similarities, double_similarities = get_rankings_with_threshold(data, data.query_ids, threshold, device, top_k,n_examples)

    all_rankings_test, _, _ = get_rankings_with_threshold(data_test, data_test.query_ids, threshold, device, top_k,n_examples)


    # Create output directory if it doesn't exist
    output_dir = "data_chunks_cos_sim"
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON files
    # output_file = os.path.join(output_dir, "cosine_sim_rank_threshold_only_single.json")
    # with open(output_file, 'w') as f:
    #     json.dump(all_rankings, f, indent=2)

    output_file = os.path.join(output_dir, "cosine_sim_rank_retrieved_test_single.json") #cosine_sim_rank_threshold_test
    with open(output_file, 'w') as f:
        json.dump(cosine_sim_results, f, indent=2)

    # output_file = os.path.join(output_dir, "cosine_sim_rank_threshold_only_single_test.json")
    # with open(output_file, 'w') as f:
    #     json.dump(all_rankings_test, f, indent=2)

    # output_file = os.path.join(output_dir, "single_similarities.json")
    # with open(output_file, 'w') as f:
    #     json.dump(single_similarities, f, indent=2)

    # output_file = os.path.join(output_dir, "double_similarities.json")
    # with open(output_file, 'w') as f:
    #     json.dump(double_similarities, f, indent=2)

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
    
    # results_file = os.path.join(output_dir, "threshold_optimization_results.json")
    # with open(results_file, 'w') as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()