import json
import os

# Define the file paths
RL_MODEL_FILE = 'data_analysis/rl_model_retrieved_test_single.json' 
COS_SIM_FILE = 'data_analysis/cosine_sim_rank_retrieved_test_single.json'
CHUNKS_SCORES_FILE = 'data_analysis/chunks_completeness_test.json'
OUTPUT_MERGED = "data_analysis/comprehensive_merged_results_single.json"
OUTPUT_COMBINED_FILTER = "data_analysis/no_isolated_chunks_single.json"
OUTPUT_ADJACENT_ANALYSIS = "data_analysis/adjacent_chunks_analysis_single.json"

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


# --- Filtering Logic 1: Isolated Chunks ---
def has_isolated_chunks(relevant_chunks_list):
    """
    Returns True if ANY relevant chunk is isolated (no relevant neighbor N-1 or N+1).
    """
    if not relevant_chunks_list:
        return False

    relevant_ids = set()
    for item in relevant_chunks_list:
        for k in item.keys():
            try:
                relevant_ids.add(int(k))
            except ValueError:
                continue

    for chunk_id in relevant_ids:
        prev_id = chunk_id - 1
        next_id = chunk_id + 1
        
        # If BOTH neighbors are missing, this chunk is isolated
        if (prev_id not in relevant_ids) and (next_id not in relevant_ids):
            return True

    return False


# --- Filtering Logic 2: Long Sequences ---
def has_long_consecutive_sequence(relevant_chunks_list, max_allowed_length=3):
    """
    Returns True if the query has 4 or more consecutive relevant chunks.
    """
    if not relevant_chunks_list:
        return False

    relevant_ids = set()
    for item in relevant_chunks_list:
        for k in item.keys():
            try:
                relevant_ids.add(int(k))
            except ValueError:
                continue

    if not relevant_ids:
        return False
    
    sorted_ids = sorted(list(relevant_ids))

    if len(sorted_ids) <= max_allowed_length:
         return False

    max_consecutive_count = 0
    current_consecutive_count = 1

    for i in range(1, len(sorted_ids)):
        if sorted_ids[i] == sorted_ids[i-1] + 1:
            current_consecutive_count += 1
        else:
            current_consecutive_count = 1
        
        max_consecutive_count = max(max_consecutive_count, current_consecutive_count)
        
        if max_consecutive_count > max_allowed_length:
            return True

    return max_consecutive_count > max_allowed_length


# --- Combined Analysis Function ---
def process_combined_analysis(full_data):
    print("-" * 30)
    print("Starting Analysis: Combining both filters...")

    filtered_data = []
    
    # Accumulators for averages
    rl_scores = []
    cos_scores = []

    for entry in full_data:
        chunks = entry.get('relevant_chunks', [])
        
        is_isolated = has_isolated_chunks(chunks)
        has_long_seq = has_long_consecutive_sequence(chunks)

        # KEEP the query if it is NOT isolated AND does NOT have a long sequence
        if not is_isolated and not has_long_seq:
            filtered_data.append(entry)
            
            # Collect scores
            if entry.get('rl_f1_score') is not None:
                rl_scores.append(entry['rl_f1_score'])
            if entry.get('cos_sim_f1_score') is not None:
                cos_scores.append(entry['cos_sim_f1_score'])

    count = len(filtered_data)
    print(f"Total Queries (Merged):                       {len(full_data)}")
    print(f"NO isolated chunks and NO long sequences:     {count}")

    if count > 0:
        avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0
        avg_cos = sum(cos_scores) / len(cos_scores) if cos_scores else 0
        
        print("-" * 30)
        print(f"Average RL F1 Score:       {avg_rl:.4f}")
        print(f"Average CosSim F1 Score:   {avg_cos:.4f}")
        print("-" * 30)
    else:
        print("No queries matched the criteria.")

    return filtered_data



def calculate_f1_score(retrieved_chunks, relevant_chunks_ids):
    """
    Computes the F1 score (harmonic mean of Precision and Recall)
    based on the sets of retrieved and relevant chunk IDs.
    """
    if not relevant_chunks_ids:
        if not retrieved_chunks:
            return 1.0
        return 0.0

    retrieved_set = set(retrieved_chunks)
    relevant_set = set(relevant_chunks_ids)

    tp = len(retrieved_set.intersection(relevant_set))
    fp = len(retrieved_set.difference(retrieved_set))
    fn = len(relevant_set.difference(retrieved_set))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def filter_and_recalculate_f1_summary(filtered_results):
    """
    Filters the retrieved chunk IDs using the new rules (proximity filter + 
    new adjacent score-based filter), recalculates F1 scores for each query, 
    and prints the average new F1 scores across all queries.

    Args:
        filtered_results (list): A list of dictionaries containing query results.

    Returns:
        list: The modified list of results with updated retrieved lists and F1 scores.
    """
    total_rl_f1_new = 0.0
    total_cos_sim_f1_new = 0.0
    num_queries = len(filtered_results)
    
    if num_queries == 0:
        print("The filtered_results list is empty. No F1 scores to compute.")
        return []

    for result in filtered_results:
        # 1. Prepare relevant chunk data
        # relevant_data is a dictionary mapping ID (int) to score (float)
        relevant_data = {}
        for item in result['relevant_chunks']:
            chunk_id_str = list(item.keys())[0]
            score = list(item.values())[0]
            relevant_data[int(chunk_id_str)] = score
        
        relevant_ids_sorted = sorted(relevant_data.keys())
        
        # 2. Filtering: Step A - Filter by Proximity to Relevant Chunks
        # This step remains the same: keep IDs that are relevant, relevant-1, or relevant+1.
        proximity_set = set()
        for chunk_id in relevant_ids_sorted:
            proximity_set.add(chunk_id)
            proximity_set.add(chunk_id - 1)
            proximity_set.add(chunk_id + 1)
            
        result['rl_model_retrieved'] = [
            chunk_id for chunk_id in result['rl_model_retrieved'] 
            if chunk_id in proximity_set
        ]
        result['cos_sim_retrieved_chunks'] = [
            chunk_id for chunk_id in result['cos_sim_retrieved_chunks'] 
            if chunk_id in proximity_set
        ]

        # 3. Filtering: Step B - Apply New Adjacent Chunk Rules
        ids_to_remove = set()
        i = 0
        while i < len(relevant_ids_sorted):
            id_a = relevant_ids_sorted[i]
            score_a = relevant_data[id_a]
            
            # Check for potential adjacent chunks (consecutive IDs in the *original* list)
            
            # Look ahead one step
            if i + 1 < len(relevant_ids_sorted):
                id_b = relevant_ids_sorted[i+1]
                
                # Check for *two* adjacent chunks: id_a and id_b are neighbors
                if id_b == id_a + 1:
                    
                    # Look ahead two steps
                    if i + 2 < len(relevant_ids_sorted):
                        id_c = relevant_ids_sorted[i+2]
                        
                        # Case 2: Three adjacent chunks (a, b, c)
                        if id_c == id_b + 1:
                            score_b = relevant_data[id_b]
                            score_c = relevant_data[id_c]
                            
                            # Find the one with the highest score
                            max_score = max(score_a, score_b, score_c)
                            
                            if score_a == max_score: ids_to_remove.add(id_a)
                            elif score_b == max_score: ids_to_remove.add(id_b)
                            elif score_c == max_score: ids_to_remove.add(id_c)
                            
                            # If scores are tied, any one of the tied max-score IDs can be removed.
                            # We'll just take the first one found with the max score.
                            
                            # Skip ahead 3 positions since these 3 chunks have been processed
                            i += 3
                            continue
                    
                    # Case 1: Two adjacent chunks (a, b)
                    else:
                        score_b = relevant_data[id_b]
                        
                        # Keep only the one with the lower score
                        if score_a < score_b: ids_to_remove.add(id_b)
                        else: ids_to_remove.add(id_a) 
                        
                        # If scores are tied, the rule is ambiguous. We'll remove the later ID (id_b) for determinism.
                        # Note: The code above covers the tie-breaker by removing 'b' if scores are equal.

                    # Skip ahead 2 positions since these 2 chunks have been processed
                    i += 2
                    continue
            
            # Case 3: Single chunk or non-adjacent chunk - no removal rule applies
            i += 1


        # Remove the identified IDs from all three lists
        for id_to_remove in ids_to_remove:
            # Remove from relevant_chunks
            result['relevant_chunks'] = [
                item for item in result['relevant_chunks'] 
                if list(item.keys())[0] != str(id_to_remove)
            ]
            
            # Remove from retrieved lists
            result['rl_model_retrieved'] = [
                chunk_id for chunk_id in result['rl_model_retrieved'] 
                if chunk_id != id_to_remove
            ]
            result['cos_sim_retrieved_chunks'] = [
                chunk_id for chunk_id in result['cos_sim_retrieved_chunks'] 
                if chunk_id != id_to_remove
            ]
            
        # 4. Compute and Accumulate New F1 Scores
        
        new_relevant_ids = [
            int(list(item.keys())[0]) for item in result['relevant_chunks']
        ]

        new_rl_f1 = calculate_f1_score(
            result['rl_model_retrieved'], 
            new_relevant_ids
        )
        new_cos_sim_f1 = calculate_f1_score(
            result['cos_sim_retrieved_chunks'], 
            new_relevant_ids
        )

        result['rl_f1_score_new'] = new_rl_f1
        result['cos_sim_f1_score_new'] = new_cos_sim_f1
        
        total_rl_f1_new += new_rl_f1
        total_cos_sim_f1_new += new_cos_sim_f1


    ## 📊 Final Summary ##
    
    avg_rl_f1 = total_rl_f1_new / num_queries
    avg_cos_sim_f1 = total_cos_sim_f1_new / num_queries

    print("--- F1 Score Recalculation Summary (New Rules) ---")
    print(f"Total Queries Processed: **{num_queries}**")
    print(f"Average New RL Model F1 Score: **{avg_rl_f1:.4f}**")
    print(f"Average New Cos Sim F1 Score: **{avg_cos_sim_f1:.4f}**")
    
    return filtered_results




if __name__ == "__main__":
    # 1. Merge Data (Keep previous logic for the comprehensive file)
    final_data = merge_json_data()
    
    os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)
    with open(OUTPUT_MERGED, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Original merged data saved to {OUTPUT_MERGED}")

    # 2. Apply Combined Filter and Analyze
    filtered_results = process_combined_analysis(final_data)

    # 3. Save New Filtered Export
    if filtered_results:
        os.makedirs(os.path.dirname(OUTPUT_COMBINED_FILTER), exist_ok=True)
        with open(OUTPUT_COMBINED_FILTER, 'w') as f:
            json.dump(filtered_results, f, indent=4)
        print(f"Combined filter results saved to {OUTPUT_COMBINED_FILTER}")
    else:
        print("No results matched all criteria; file not created.")


    focus_d = filter_and_recalculate_f1_summary(filtered_results)


    if focus_d:
        with open("data_analysis/no_isolated_chunks_single_on_adj.json", 'w') as f:
            json.dump(focus_d, f, indent=4)