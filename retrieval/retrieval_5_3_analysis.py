import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Define the file paths
RL_MODEL_FILE = 'data_5_analysis/rl_model_retrieved_test_single.json' 
COS_SIM_FILE = 'data_4_cos_sim/cosine_sim_rank_retrieved_test_single.json'
RANKINGS = 'data_4_cos_sim/cosine_sim_rank_threshold_only_single_test.json'
CHUNKS_SCORES_FILE = 'data_2_chunk_and_label/relevant_chunks_test.json'
OUTPUT_MERGED = "data_5_analysis/comprehensive_merged_results_single.json"

def f1_score(pred_ids, true_ids, all_ids):
    tp = len(set(pred_ids) & set(true_ids))
    fp = len(set(pred_ids) - set(true_ids))
    fn = len(set(true_ids) - set(pred_ids))
    tn = all_ids - (tp + fp + fn)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0 #tpr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return f1, recall, precision, fpr


def merge_json_data():
    """
    Merge data from three JSON files by matching queries.
    The resulting merged data is ordered by page_id and query.
    """
    
    # Load all data files
    with open(RL_MODEL_FILE, 'r') as f:
        rl_data = json.load(f)
    
    with open(COS_SIM_FILE, 'r') as f:
        cos_sim_data = json.load(f)
    
    with open(CHUNKS_SCORES_FILE, 'r') as f:
        chunks_data = json.load(f)

    with open(RANKINGS, 'r') as f:
        rankings = json.load(f)
    
    # Create lookup dictionaries
    rl_lookup = {item['query']: item for item in rl_data}
    cos_sim_lookup = {item['query_desc']: item for item in cos_sim_data}
    chunks_count_lookup = {v["query_desc"]: v["total_chunks_count"] for v in rankings.values()}
    chunks_lookup = defaultdict(list)
    
    for item in chunks_data:
        key = item['query']
        chunks_lookup[key].append(item)
    
    # Merge data
    merged_results = []
    seen_queries = set()
    
    # Iterate through chunks data as the base (contains page_id and query)
    for item in chunks_data:
        query = item['query']
        page_id = item['page_id']
        
        # Skip duplicates (in case same query appears multiple times)
        query_page_key = (query, page_id)
        if query_page_key in seen_queries:
            continue
        seen_queries.add(query_page_key)
        
        # Get corresponding data from RL and cosine similarity
        rl_item = rl_lookup.get(query)
        cos_sim_item = cos_sim_lookup.get(query)
        chunks_count_item = chunks_count_lookup[query]
        
        # Build merged object
        merged_obj = {
            "page_id": page_id,
            "query": query,
            "rl_f1_score": rl_item['rl_f1_score'] if rl_item else None,
            "cos_sim_f1_score": cos_sim_item['f1_score'] if cos_sim_item else None,
            "rl_model_retrieved": rl_item['rl_model_retrieved'] if rl_item else [],
            "cos_sim_retrieved_chunks": cos_sim_item['cos_sim_retrieved_chunks'] if cos_sim_item else [],
            "relevant_chunks": item['relevant_chunks'],
            "relevant_paragraph_origin": item['relevant_paragraph_origin'],
            "chunk_relevant_portion": item['chunk_relevant_portion'],
            "total_chunks_count": chunks_count_item
        }
        
        merged_results.append(merged_obj)
    
    # Sort by page_id, then by query
    merged_results.sort(key=lambda x: (x['page_id'], x['query']))
    
    return merged_results


def calculate_metrics(merged_data):
    """
    Calculate average F1 scores and count of queries.
    """
    rl_scores = []
    cos_sim_scores = []
    
    for item in merged_data:
        if item['rl_f1_score'] is not None:
            rl_scores.append(item['rl_f1_score'])
        if item['cos_sim_f1_score'] is not None:
            cos_sim_scores.append(item['cos_sim_f1_score'])
    
    avg_rl_f1 = sum(rl_scores) / len(rl_scores) if rl_scores else 0
    avg_cos_sim_f1 = sum(cos_sim_scores) / len(cos_sim_scores) if cos_sim_scores else 0
    query_count = len(merged_data)
    
    return {
        'avg_rl_f1_score': avg_rl_f1,
        'avg_cos_sim_f1_score': avg_cos_sim_f1,
        'query_count': query_count
    }


def process_all_queries(final_data, completeness_threshold, topk, return_avg_scores = False):

    filtered_results = []
    rl_scores = []
    rl_recall_scores = []
    rl_precision_scores = []
    rl_fpr_scores = []
    cos_sim_scores = []
    cos_sim_recall_scores = []
    cos_sim_precision_scores = []
    cos_sim_fpr_scores = []


    for item in final_data:
        query = item['query']
        
        # Identify keys in chunk_relevant_portion above threshold
        keys_to_remove = set()
        for chunk_dict in item['chunk_relevant_portion']:
            for key, value in chunk_dict.items():
                if value >= completeness_threshold:
                    keys_to_remove.add(key)
        
        # Remove identified keys from chunk_relevant_portion and relevant_paragraph_origin
        chunk_relevant_portion_filtered = [
            {k: v for k, v in chunk_dict.items() if k not in keys_to_remove}
            for chunk_dict in item['chunk_relevant_portion']
        ]
        chunk_relevant_portion_filtered = [d for d in chunk_relevant_portion_filtered if d]
        
        relevant_paragraph_origin_filtered = {
            k: v for k, v in item['relevant_paragraph_origin'].items()
            if k not in keys_to_remove
        }

        chunks_to_remove = [float(x) for x in keys_to_remove]

        remaining_chunks = set(int(k) for k in relevant_paragraph_origin_filtered.keys())
        
        # Filter the lists
        rl_model_retrieved_filtered = [
            chunk_id for chunk_id in item['rl_model_retrieved']
            if chunk_id not in chunks_to_remove
        ][:topk]
        
        cos_sim_retrieved_chunks_filtered = [
            chunk_id for chunk_id in item['cos_sim_retrieved_chunks']
            if chunk_id not in chunks_to_remove
        ][:topk]
        
        relevant_chunks_filtered = [
            chunk_id for chunk_id in item['relevant_chunks']
            if chunk_id not in chunks_to_remove
        ]

        if not relevant_chunks_filtered:
            continue
                
        rl_f1, rl_recall, rl_precision, rl_fpr = f1_score(rl_model_retrieved_filtered, relevant_chunks_filtered, item['total_chunks_count'])
        cos_sim_f1, cos_sim_recall, cos_sim_precision, cos_sim_fpr = f1_score(cos_sim_retrieved_chunks_filtered, relevant_chunks_filtered, item['total_chunks_count'])
        
        rl_scores.append(rl_f1)
        rl_recall_scores.append(rl_recall)
        rl_precision_scores.append(rl_precision)
        rl_fpr_scores.append(rl_fpr)

        cos_sim_scores.append(cos_sim_f1)
        cos_sim_recall_scores.append(cos_sim_recall)
        cos_sim_precision_scores.append(cos_sim_precision)
        cos_sim_fpr_scores.append(cos_sim_fpr)
        
        # Create filtered item
        filtered_item = {
            "page_id": item['page_id'],
            "query": item['query'],
            "rl_f1_score": rl_f1,
            "cos_sim_f1_score": cos_sim_f1,
            "rl_model_retrieved": rl_model_retrieved_filtered,
            "cos_sim_retrieved_chunks": cos_sim_retrieved_chunks_filtered,
            "relevant_chunks": relevant_chunks_filtered,
            "relevant_paragraph_origin": relevant_paragraph_origin_filtered,
            "chunk_relevant_portion": chunk_relevant_portion_filtered
        }
        
        filtered_results.append(filtered_item)
    
    # Calculate Averages for RL
    rl_f1_avg = sum(rl_scores) / len(rl_scores)
    rl_rec_avg = sum(rl_recall_scores) / len(rl_recall_scores)
    rl_precision_avg = sum(rl_precision_scores) / len(rl_precision_scores)
    rl_fpr_avg = sum(rl_fpr_scores) / len(rl_fpr_scores)

    # Calculate Averages for Cos-Sim
    cs_f1_avg = sum(cos_sim_scores) / len(cos_sim_scores)
    cs_rec_avg = sum(cos_sim_recall_scores) / len(cos_sim_recall_scores)
    cs_precision_avg = sum(cos_sim_precision_scores) / len(cos_sim_precision_scores)
    cs_fpr_avg = sum(cos_sim_fpr_scores) / len(cos_sim_fpr_scores)

    if return_avg_scores:
        return rl_f1_avg, rl_rec_avg, rl_precision_avg, cs_f1_avg, cs_rec_avg, cs_precision_avg

    print(f"\nProcessing Complete!")
    print(f"Total queries: {len(final_data)}")
    print(f"Queries retained: {len(filtered_results)}")
    print(f"Queries dropped: {len(final_data) - len(filtered_results)}")

    # RL Metrics
    print(f"\n--- RL Model Metrics ---")
    print(f"Average RL F1 Score:      {rl_f1_avg:.4f}")
    print(f"Average RL Recall (TPR):  {rl_rec_avg:.4f}")
    print(f"Average RL Precision:     {rl_precision_avg:.4f}")
    print(f"Average RL FPR:           {rl_fpr_avg:.4f}")

    # Cos-Sim Metrics
    print(f"\n--- Cos-Sim Metrics ---")
    print(f"Average CS F1 Score:     {cs_f1_avg:.4f}")
    print(f"Average CS Recall (TPR): {cs_rec_avg:.4f}")
    print(f"Average CS Precision:    {cs_precision_avg:.4f}")
    print(f"Average CS FPR:          {cs_fpr_avg:.4f}")

    # Variation (using F1 as the primary comparison)
    variation = ((rl_f1_avg - cs_f1_avg) / cs_f1_avg) * 100
    print(f"\nF1 Score Variation: {(rl_f1_avg - cs_f1_avg):.4f}, {variation:.4f}%")

    return filtered_results

def plot_relevance_coverage(thresholds, final_data):

    rl_recall, cs_recall = [], []

    for t in thresholds:
        rl_f1, rl_rec, rl_prec, cs_f1, cs_rec, cs_prec = process_all_queries(
            final_data, t, topk=100, return_avg_scores=True
        )
        rl_recall.append(rl_rec)
        cs_recall.append(cs_rec)

    thresholds_arr = np.array(thresholds)

    # Fit straight lines
    rl_fit = np.poly1d(np.polyfit(thresholds_arr, rl_recall, 1))
    cs_fit = np.poly1d(np.polyfit(thresholds_arr, cs_recall, 1))

    fig, ax = plt.subplots(figsize=(6, 5))

    # Data series
    ax.plot(thresholds_arr, rl_recall,
            color="steelblue", marker="o", linewidth=2, markersize=6, label="RL")
    ax.plot(thresholds_arr, cs_recall,
            color="tomato", marker="s", linewidth=2, markersize=6,
            linestyle="--", label="Cosine Sim")

    # Trend lines
    ax.plot(thresholds_arr, rl_fit(thresholds_arr),
            color="steelblue", linewidth=1.5, linestyle=":", alpha=0.8,
            label=f"RL trend (slope={rl_fit.c[0]:+.3f})")
    ax.plot(thresholds_arr, cs_fit(thresholds_arr),
            color="tomato", linewidth=1.5, linestyle=":", alpha=0.8,
            label=f"CS trend (slope={cs_fit.c[0]:+.3f})")

    ax.set_xlabel("Completeness Threshold", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Recall: RL vs Cosine Similarity over Relevance Coverage C")
    ax.set_xticks(thresholds_arr)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    plt.savefig("recall_over_threshold.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → recall_over_threshold.png")

def analyze():
    # Merge data
    final_data = merge_json_data()
    
    # Save merged data to file
    os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)
    with open(OUTPUT_MERGED, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Merged data saved to {OUTPUT_MERGED}")
    
    # Calculate and print metrics
    metrics = calculate_metrics(final_data)
    print(f"\n{'='*50}")
    print(f"Metrics Summary:")
    print(f"{'='*50}")
    print(f"Total number of queries: {metrics['query_count']}")
    print(f"Average RL F1 Score: {metrics['avg_rl_f1_score']:.6f}")
    print(f"Average Cosine Similarity F1 Score: {metrics['avg_cos_sim_f1_score']:.6f}")
    print(f"{'='*50}")

    completeness_threshold = 1 # 0.8 means chunks with more than 80% relevant are removed

    filtered_results = process_all_queries(final_data, completeness_threshold, topk = 100)

    filtered_results = process_all_queries(final_data, 1, topk = 100, return_avg_scores=True)

    with open('data_5_analysis/filtered_results.json', 'w') as f:
        json.dump(filtered_results, f, indent=2)

    thresholds = np.arange(1, 0, -0.1)

    plot_relevance_coverage(thresholds, final_data)