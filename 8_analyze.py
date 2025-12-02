import json
import os

# Define the file paths
RL_MODEL_FILE = 'data_analysis/rl_model_retrieved_test.json'
COS_SIM_FILE = 'data_analysis/cosine_sim_rank_retrieved_test.json'
CHUNKS_SCORES_FILE = 'data_analysis/chunks_completeness_test.json'
OUTPUT_MERGED = "data_analysis/comprehensive_merged_results.json"
OUTPUT_COMBINED_FILTER = "data_analysis/no_isolated_chunks.json" # User specified output filename
OUTPUT_ADJACENT_ANALYSIS = "data_analysis/adjacent_chunks_analysis.json"

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


# --- NEW: Adjacent Chunks Analysis ---
def get_chunk_scores_dict(relevant_chunks_list):
    """Extract chunk IDs and their relevance scores into a dictionary."""
    scores = {}
    for item in relevant_chunks_list:
        for k, v in item.items():
            try:
                chunk_id = int(k)
                scores[chunk_id] = float(v)
            except (ValueError, TypeError):
                continue
    return scores


def identify_adjacent_chunks(chunk_scores):
    """
    Identify adjacent chunks that are next to a more relevant chunk.
    Returns a set of adjacent chunk IDs.
    """
    if not chunk_scores:
        return set()
    
    sorted_chunks = sorted(chunk_scores.items())
    adjacent = set()
    
    for i, (chunk_id, score) in enumerate(sorted_chunks):
        # Check previous neighbor
        if i > 0:
            prev_id, prev_score = sorted_chunks[i-1]
            if prev_id == chunk_id - 1 and prev_score > score:
                adjacent.add(chunk_id)
        
        # Check next neighbor
        if i < len(sorted_chunks) - 1:
            next_id, next_score = sorted_chunks[i+1]
            if next_id == chunk_id + 1 and next_score > score:
                adjacent.add(chunk_id)
    
    return adjacent


def get_retrieved_chunks_set(retrieved_list):
    """Extract chunk IDs from retrieved chunks list."""
    chunks = set()
    if not retrieved_list:
        return chunks
    
    for item in retrieved_list:
        if isinstance(item, dict):
            for k in item.keys():
                try:
                    chunks.add(int(k))
                except (ValueError, TypeError):
                    continue
        elif isinstance(item, (int, str)):
            try:
                chunks.add(int(item))
            except (ValueError, TypeError):
                continue
    
    return chunks


def analyze_adjacent_chunks(filtered_data):
    """
    Analyze adjacent chunks errors between CosSim and RL models.
    """
    print("\n" + "=" * 60)
    print("ADJACENT CHUNKS ANALYSIS")
    print("=" * 60)
    
    results = []
    
    # Metrics for CosSim errors that RL fixed
    cos_fp_total = 0  # False positives (took adjacent when shouldn't)
    cos_fp_rl_correct = 0  # RL correctly didn't take it
    
    cos_fn_total = 0  # False negatives (didn't take adjacent when should)
    cos_fn_rl_correct = 0  # RL correctly took it
    
    # Metrics for RL errors that CosSim fixed
    rl_fp_total = 0
    rl_fp_cos_correct = 0
    
    rl_fn_total = 0
    rl_fn_cos_correct = 0
    
    for entry in filtered_data:
        query = entry.get('query', '')
        chunks_list = entry.get('relevant_chunks', [])
        
        chunk_scores = get_chunk_scores_dict(chunks_list)
        adjacent_chunks = identify_adjacent_chunks(chunk_scores)
        
        if not adjacent_chunks:
            continue
        
        all_relevant = set(chunk_scores.keys())
        non_adjacent_relevant = all_relevant - adjacent_chunks
        
        rl_retrieved = get_retrieved_chunks_set(entry.get('rl_model_retrieved', []))
        cos_retrieved = get_retrieved_chunks_set(entry.get('cos_sim_retrieved_chunks', []))
        
        # Analyze each adjacent chunk
        query_result = {
            'query': query,
            'adjacent_chunks': sorted(list(adjacent_chunks)),
            'cos_errors': [],
            'rl_errors': []
        }
        
        for adj_chunk in adjacent_chunks:
            cos_took = adj_chunk in cos_retrieved
            rl_took = adj_chunk in rl_retrieved
            
            # Adjacent chunks should NOT be taken (they're less relevant)
            should_take = False
            
            # CosSim false positive (took when shouldn't)
            if cos_took and not should_take:
                cos_fp_total += 1
                if not rl_took:
                    cos_fp_rl_correct += 1
                    query_result['cos_errors'].append({
                        'chunk_id': adj_chunk,
                        'error_type': 'false_positive',
                        'rl_correct': True
                    })
                else:
                    query_result['cos_errors'].append({
                        'chunk_id': adj_chunk,
                        'error_type': 'false_positive',
                        'rl_correct': False
                    })
            
            # RL false positive (took when shouldn't)
            if rl_took and not should_take:
                rl_fp_total += 1
                if not cos_took:
                    rl_fp_cos_correct += 1
                    query_result['rl_errors'].append({
                        'chunk_id': adj_chunk,
                        'error_type': 'false_positive',
                        'cos_correct': True
                    })
                else:
                    query_result['rl_errors'].append({
                        'chunk_id': adj_chunk,
                        'error_type': 'false_positive',
                        'cos_correct': False
                    })
        
        if query_result['cos_errors'] or query_result['rl_errors']:
            results.append(query_result)
    
    # Print summary
    print(f"\nCosSim Errors on Adjacent Chunks:")
    print(f"  False Positives (took adjacent chunks): {cos_fp_total}")
    if cos_fp_total > 0:
        pct = (cos_fp_rl_correct / cos_fp_total) * 100
        print(f"  RL Model corrected: {cos_fp_rl_correct} ({pct:.2f}%)")
    
    print(f"\nRL Model Errors on Adjacent Chunks:")
    print(f"  False Positives (took adjacent chunks): {rl_fp_total}")
    if rl_fp_total > 0:
        pct = (rl_fp_cos_correct / rl_fp_total) * 100
        print(f"  CosSim corrected: {rl_fp_cos_correct} ({pct:.2f}%)")
    
    print("=" * 60)
    
    summary = {
        'cosSim_errors': {
            'false_positives': cos_fp_total,
            'rl_corrected': cos_fp_rl_correct,
            'rl_correction_rate': (cos_fp_rl_correct / cos_fp_total * 100) if cos_fp_total > 0 else 0
        },
        'rl_errors': {
            'false_positives': rl_fp_total,
            'cos_corrected': rl_fp_cos_correct,
            'cos_correction_rate': (rl_fp_cos_correct / rl_fp_total * 100) if rl_fp_total > 0 else 0
        },
        'detailed_results': results
    }
    
    return summary


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
    
    # 4. Analyze Adjacent Chunks
    if filtered_results:
        adjacent_analysis = analyze_adjacent_chunks(filtered_results)
        
        os.makedirs(os.path.dirname(OUTPUT_ADJACENT_ANALYSIS), exist_ok=True)
        with open(OUTPUT_ADJACENT_ANALYSIS, 'w') as f:
            json.dump(adjacent_analysis, f, indent=4)
        print(f"\nAdjacent chunks analysis saved to {OUTPUT_ADJACENT_ANALYSIS}")