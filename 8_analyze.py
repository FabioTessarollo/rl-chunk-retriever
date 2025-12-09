import json
import os
from collections import defaultdict


# Define the file paths
RL_MODEL_FILE = 'data_analysis/rl_model_retrieved_test_single.json' 
COS_SIM_FILE = 'data_chunks_cos_sim/cosine_sim_rank_retrieved_test_single.json'
CHUNKS_SCORES_FILE = 'data_chunks/relevant_chunks_test.json'
OUTPUT_MERGED = "data_analysis/comprehensive_merged_results_single.json"

def f1_score(pred_ids, true_ids):
    tp = len(set(pred_ids) & set(true_ids))
    fp = len(set(pred_ids) - set(true_ids))
    fn = len(set(true_ids) - set(pred_ids))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


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
    
    # Create lookup dictionaries
    rl_lookup = {item['query']: item for item in rl_data}
    cos_sim_lookup = {item['query_desc']: item for item in cos_sim_data}
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
            "chunk_relevant_portion": item['chunk_relevant_portion']
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


def process_all_queries(final_data, completeness_threshold):

    filtered_results = []
    rl_scores = []
    cos_sim_scores = []

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
        ]
        
        cos_sim_retrieved_chunks_filtered = [
            chunk_id for chunk_id in item['cos_sim_retrieved_chunks']
            if chunk_id not in chunks_to_remove
        ]
        
        relevant_chunks_filtered = [
            chunk_id for chunk_id in item['relevant_chunks']
            if chunk_id not in chunks_to_remove
        ]

        if not relevant_chunks_filtered:
            continue
                
        rl_f1 = f1_score(rl_model_retrieved_filtered, relevant_chunks_filtered)
        cos_sim_f1 = f1_score(cos_sim_retrieved_chunks_filtered, relevant_chunks_filtered)
        
        rl_scores.append(rl_f1)
        cos_sim_scores.append(cos_sim_f1)
        
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
    
    rl_f1_avg = sum(rl_scores) / len(rl_scores)
    cs_f1_avg = sum(cos_sim_scores) / len(cos_sim_scores)

    print(f"\nProcessing Complete!")
    print(f"Total queries: {len(final_data)}")
    print(f"Queries retained: {len(filtered_results)}")
    print(f"Queries dropped: {len(final_data) - len(filtered_results)}")
    print(f"\nAverage RL F1 Score: {rl_f1_avg:.4f}" if rl_scores else "No queries retained")
    print(f"Average Cos-Sim F1 Score: {cs_f1_avg:.4f}" if cos_sim_scores else "No queries retained")
    print(f"Variation: {((rl_f1_avg - cs_f1_avg) / cs_f1_avg)*100:.4f}%")

    return filtered_results

if __name__ == "__main__":
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

    completeness_threshold = 0.3

    filtered_results = process_all_queries(final_data, completeness_threshold)

    with open('data_analysis/filtered_results.json', 'w') as f:
        json.dump(filtered_results, f, indent=2)