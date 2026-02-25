import random
import json
import os
import numpy as np
import torch
from retrieval.Data import Data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import auc

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

    # ROC - Lists for ROC (true/false labels per chunk)
    all_true_labels = []
    all_pred_labels = []

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

        # ROC - Build per‑chunk labels for ROC
        all_true_labels.extend([1 if cid in relevant_chunks else 0 for cid in page])
        all_pred_labels.extend([1 if cid in selected_chunks else 0 for cid in page])

        # Add query result to list
        cosine_sim_results.append({
            "query_desc": query_desc,
            "f1_score": f1_score,
            "cos_sim_retrieved_chunks": list(selected_chunks)
        })
    
    avg_recall = recall_total / n if n > 0 else 0
    avg_precision = precision_total / n if n > 0 else 0
    avg_f1_score = f1_score_total / n if n > 0 else 0

    # ROC - Convert to numpy for ROC math
    y_true = np.array(all_true_labels, dtype=int)
    y_pred = np.array(all_pred_labels, dtype=int)

    # ROC - Count TP/FP/TN/FN (added part)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    per_chunk_stats = {"tp": int(tp), "fp": int(fp),
                       "tn": int(tn), "fn": int(fn)}
    
    return avg_recall, avg_precision, avg_f1_score, cosine_sim_results, per_chunk_stats

def find_optimal_threshold(data, training_query_ids, device, threshold_range=(0.77, 0.83), step=0.01, do_roc = False):
    """Find the optimal cosine similarity threshold using training data"""
    best_threshold = 0.0
    best_f1_score = 0.0
    
    print("Finding optimal threshold on training data...")
    thresholds = np.concatenate(([0], np.arange(threshold_range[0], threshold_range[1] + step, step), [1]))
    roc_points = []
    
    for threshold in thresholds:
        _, _, f1_score, _, per_chunk_stats = evaluate_with_threshold(data, training_query_ids, threshold, device)
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
        
        print(f"Threshold: {threshold:.3f}, F1: {f1_score:.4f}")

        # ROC
        tp, fp, fn, tn = per_chunk_stats["tp"], per_chunk_stats["fp"], \
                         per_chunk_stats["fn"], per_chunk_stats["tn"]
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        roc_points.append((threshold, tpr, fpr))

    # ROC
    if do_roc:
        roc_points_sorted = sorted(roc_points, key=lambda x: x[2])  # Sort by FPR
        thr_vals, tprs, fprs = zip(*roc_points_sorted)
        roc_auc = auc(fprs, tprs)

        plt.figure(figsize=(6,5))
        plt.plot(fprs, tprs, marker='o', label=f'ROC (AUC={roc_auc:.3f})')
        plt.plot([0,1], [0,1], 'k--', label='Chance')
        
        # Add threshold labels above each point
        for i, (fpr, tpr, thr) in enumerate(zip(fprs, tprs, thr_vals)):
            plt.annotate(f'{thr:.2f}', 
                        xy=(fpr, tpr), 
                        xytext=(0, 8),  # 8 points above the point
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        alpha=0.7)
        
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve – Threshold Sweep')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('ROC-AUC.png')
    
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
        total_chunks_count = len(page)
        
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
            "relevant_chunks": [chunk_id for chunk_id, _ in top_chunks.items()],
            "total_chunks_count" : total_chunks_count
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

    train_set, validation_set = data.balanced_split_query_ids(data.query_ids, 0.6)

    pages_path_test = f"data_3_embed/pages_chunked_emb_test.json"
    relevant_path_test = f"data_3_embed/relevant_chunks_emb_test.json"

    data_test = Data(pages_path_test, relevant_path_test)
    data_test.load_pages()
    data_test.load_relevant()

    n_examples = 2


    # Evaluate on training set with optimal threshold
    optimal_threshold = find_optimal_threshold(data, train_set, device, (0.76, 0.86), do_roc=True)
    
    # Evaluate on validation set with optimal threshold
    print(f"\n=== Validation Set Results (Threshold: {optimal_threshold:.3f}) ===")
    val_recall, val_precision, val_f1, _, _= evaluate_with_threshold(data, validation_set, optimal_threshold, device)
    print(f"Recall: {val_recall:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Find optimal threshold using full training data
    optimal_threshold = find_optimal_threshold(data, data.query_ids, device, (0.76, 0.86))

    # Evaluate on test set with optimal threshold
    print(f"\n=== Test Set Results (Threshold: {optimal_threshold:.3f}) ===")
    train_recall, train_precision, train_f1, cosine_sim_results, _ = evaluate_with_threshold(data_test, data_test.query_ids, optimal_threshold, device)
    print(f"Recall: {train_recall:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"F1 Score: {train_f1:.4f}")
    
    # Get rankings for all queries using a threshold
    top_k = 40 # max rank size
    threshold = 0.77 # min similarity
    all_rankings_train = get_rankings_with_threshold(data, data.query_ids, threshold, device, top_k,n_examples)
    all_rankings_test = get_rankings_with_threshold(data_test, data_test.query_ids, threshold, device, top_k,n_examples)

    output_dir = "data_4_cos_sim"
    os.makedirs(output_dir, exist_ok=True)

    # Save train set rankings for RL model train and inference
    # output_file = os.path.join(output_dir, f"cosine_sim_rank_threshold_only_single_train.json")
    # with open(output_file, 'w') as f:
    #     json.dump(all_rankings_train, f, indent=2)

    # # Save test set rankings for RL model inference
    # output_file = os.path.join(output_dir, f"cosine_sim_rank_threshold_only_single_test.json")
    # with open(output_file, 'w') as f:
    #     json.dump(all_rankings_test, f, indent=2)

    # # Save TEST cos sim results for analysis
    # output_file = os.path.join(output_dir, f"cosine_sim_rank_retrieved_test_single.json")
    # with open(output_file, 'w') as f:
    #     json.dump(cosine_sim_results, f, indent=2)


