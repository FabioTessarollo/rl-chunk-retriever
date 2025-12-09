import json
import os
from sklearn.metrics import f1_score
from collections import defaultdict

# Define the file paths
RL_MODEL_FILE = 'data_analysis/rl_model_retrieved_test_single.json' 
COS_SIM_FILE = 'data_chunks_cos_sim/cosine_sim_rank_retrieved_test_single.json'
CHUNKS_SCORES_FILE = 'data_chunks/relevant_chunks_test.json'
OUTPUT_MERGED = "data_analysis/comprehensive_merged_results_single.json"

def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []

def get_ordered_entry(entry):
    """Helper to order dictionary keys based on user requirements."""
    desired_order = [
        "page_id", "query", "rl_f1_score", "cos_sim_f1_score",
        "rl_model_retrieved", "cos_sim_retrieved_chunks", "relevant_chunks"
    ]
    ordered_entry = {}
    for key in desired_order:
        if key in entry:
            ordered_entry[key] = entry[key]
    # Append remaining keys
    for key, value in entry.items():
        if key not in ordered_entry:
            ordered_entry[key] = value
    return ordered_entry

def merge_json_data():
    rl_data = load_json_data(RL_MODEL_FILE)
    cos_sim_data = load_json_data(COS_SIM_FILE)
    chunk_scores_data = load_json_data(CHUNKS_SCORES_FILE)

    if not rl_data and not cos_sim_data and not chunk_scores_data:
        return []

    merged_data_dict = {}

    # Process RL Data
    for item in rl_data:
        if item.get("query"):
            merged_data_dict[item["query"]] = item

    # Process Cos Sim Data
    for item in cos_sim_data:
        query = item.get("query_desc")
        if query and query in merged_data_dict:
            item["cos_sim_f1_score"] = item.pop("f1_score", None)
            merged_data_dict[query].update(item)
            if "query_desc" in merged_data_dict[query]:
                del merged_data_dict[query]["query_desc"]

    # Process Chunk Scores
    for item in chunk_scores_data:
        query = item.get("query")
        if query and query in merged_data_dict:
            merged_data_dict[query].update(item)

    # Create ordered list
    return [get_ordered_entry(entry) for entry in merged_data_dict.values()]

# claude

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
        
        # Calculate new F1 scores
        # Create binary vectors for F1 calculation
        all_chunks = sorted(list(remaining_chunks))
        
        rl_retrieved_binary = [1 if chunk in rl_model_retrieved_filtered else 0 for chunk in all_chunks]
        cos_sim_retrieved_binary = [1 if chunk in cos_sim_retrieved_chunks_filtered else 0 for chunk in all_chunks]
        relevant_binary = [1 if chunk in relevant_chunks_filtered else 0 for chunk in all_chunks]
        
        rl_f1 = f1_score(relevant_binary, rl_retrieved_binary, zero_division=0)
        cos_sim_f1 = f1_score(relevant_binary, cos_sim_retrieved_binary, zero_division=0)
        
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

    print(f"\nProcessing Complete!")
    print(f"Total queries: {len(final_data)}")
    print(f"Queries retained: {len(filtered_results)}")
    print(f"Queries dropped: {len(final_data) - len(filtered_results)}")
    print(f"\nAverage RL F1 Score: {sum(rl_scores) / len(rl_scores):.4f}" if rl_scores else "No queries retained")
    print(f"Average Cos-Sim F1 Score: {sum(cos_sim_scores) / len(cos_sim_scores):.4f}" if cos_sim_scores else "No queries retained")

    return filtered_results


if __name__ == "__main__":
    # 1. Merge Data (Keep previous logic for the comprehensive file)
    final_data = merge_json_data()
    
    os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)
    with open(OUTPUT_MERGED, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Original merged data saved to {OUTPUT_MERGED}")

    completeness_threshold = 1.1

    filtered_results = process_all_queries(final_data, completeness_threshold)

    with open('data_analysis/filtered_results.json', 'w') as f:
        json.dump(filtered_results, f, indent=2)

