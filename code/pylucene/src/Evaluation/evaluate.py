import os
from ranx import Qrels, Run, evaluate

# =============================================================================
# SEARCH ENGINE EVALUATOR (TREC Format)
# -----------------------------------------------------------------------------
# This script evaluates the performance of the search engine by comparing
# the retrieved results (Run) against the ground truth (Qrels) using 
# standard Information Retrieval metrics (MAP, MRR, NDCG).
# =============================================================================

def main():
    print("=== Starting Search Engine Evaluation ===")

    # 1. DYNAMIC PATH RESOLUTION
    # Get the directory where this script is located (.../code/pylucene/src/Evaluation)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate UP three levels to reach the 'code' folder, then DOWN to 'experiment'
    EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment"))

    # Define the exact paths for the Qrels (Ground Truth) and the Run (Our results)
    # NOTE: You might need to adjust the exact filenames once you have them!
    qrels_path = os.path.join(EXPERIMENT_DIR, "docs", "qrels_dev.txt") 
    run_path = os.path.join(EXPERIMENT_DIR, "runs", "seupd-baseline-run.txt")

    # 2. VALIDATION CHECK
    if not os.path.exists(qrels_path):
        print(f"[!] Error: Qrels file not found at {qrels_path}")
        print("    Please ensure the ground truth file is in the correct folder.")
        return
        
    if not os.path.exists(run_path):
        print(f"[!] Error: Run file not found at {run_path}")
        print("    Please run the search.py script first to generate the results.")
        return

    # 3. LOAD DATA INTO MEMORY
    print("\nLoading Qrels (Ground Truth)...")
    # 'kind="trec"' specifies that the file follows the standard TREC format 
    # (query_id, iteration, doc_id, relevance)
    qrels = Qrels.from_file(qrels_path, kind="trec")

    print("Loading Run (Search Engine Results)...")
    # (query_id, iteration, doc_id, rank, score, system_name)
    run = Run.from_file(run_path, kind="trec")

    # 4. COMPUTE METRICS
    # We evaluate using the most common metrics for CLEF/TREC tasks:
    # - map: Mean Average Precision (overall ranking quality)
    # - mrr: Mean Reciprocal Rank (how high is the first relevant document)
    # - ndcg@10: Normalized Discounted Cumulative Gain for the top 10 results
    print("Computing metrics...\n")
    metrics = ["map", "mrr", "ndcg@10"]
    results_dict = evaluate(qrels, run, metrics)

    # 5. DISPLAY OFFICIAL RESULTS
    print("=======================================")
    print("          OFFICIAL RESULTS             ")
    print("=======================================")
    for metric, score in results_dict.items():
        # Print the score formatted to 4 decimal places
        print(f" {metric.upper():<10}: {score:.4f}")
    print("=======================================\n")

if __name__ == "__main__":
    main()