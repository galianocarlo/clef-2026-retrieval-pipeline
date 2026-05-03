import os
import argparse
from tqdm import tqdm
from ranx import Qrels, Run, evaluate
# =============================================================================
# SEARCH ENGINE EVALUATOR (TREC Format)
# -----------------------------------------------------------------------------
# This script evaluates the performance of the search engine by comparing
# the retrieved results (Run) against the ground truth (Qrels) using
# standard Information Retrieval metrics (MAP, MRR, NDCG).
# =============================================================================


def _count_lines(path: str) -> int:
    """Count non-empty lines so tqdm has a known total."""
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def load_trec_run(path: str) -> Run:
    """
    TREC run loader with a tqdm progress bar.
    File format: query_id  Q0  doc_id  rank  score  run_id   (whitespace-separated)
    """
    total = _count_lines(path)
    run_dict: dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Loading run", unit="line"):
            parts = line.split()
            if len(parts) < 6:
                continue
            q_id, _, doc_id, _, score, _ = parts[:6]
            run_dict.setdefault(q_id, {})[doc_id] = float(score)
    return Run(run_dict)


def load_trec_qrels(path: str) -> Qrels:
    """
    TREC qrels loader with a tqdm progress bar.
    File format: query_id  iter  doc_id  relevance   (whitespace-separated)
    """
    total = _count_lines(path)
    qrels_dict: dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Loading qrels", unit="line"):
            parts = line.split()
            if len(parts) < 4:
                continue
            q_id, _, doc_id, rel = parts[:4]
            qrels_dict.setdefault(q_id, {})[doc_id] = int(rel)
    return Qrels(qrels_dict)

def main():
    print("=== Starting Search Engine Evaluation ===")

    # 1. DYNAMIC PATH RESOLUTION
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment"))

    # CLI:  python evaluate.py [run_file] [--split train|dev]
    # Both args are optional; defaults are run-native-analyzers.txt + dev split.
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", nargs="?",
                        default=os.path.join(EXPERIMENT_DIR, "runs", "run-native-analyzers.txt"),
                        help="Path to the TREC run file produced by search.py")
    parser.add_argument("--split", choices=["train", "dev"], default="dev",
                        help="Which split's qrels to evaluate against (must match the split used by search.py)")
    args = parser.parse_args()

    qrels_path = os.path.join(EXPERIMENT_DIR, "docs", f"qrels_{args.split}.txt")
    run_path = args.run_file
    print(f"  qrels: {qrels_path}")
    print(f"  run  : {run_path}")

    # 2. VALIDATION CHECK
    if not os.path.exists(qrels_path):
        print(f"[!] Error: Qrels file not found at {qrels_path}")
        print("    Please ensure the ground truth file is in the correct folder.")
        return
        
    if not os.path.exists(run_path):
        print(f"[!] Error: Run file not found at {run_path}")
        print("    Please run the search.py script first to generate the results.")
        return

    # 3. LOAD DATA INTO MEMORY (with tqdm progress bars; ranx's own from_file has no hook)
    print("\nLoading Qrels (Ground Truth)...")
    qrels = load_trec_qrels(qrels_path)

    print("Loading Run (Search Engine Results)...")
    run = load_trec_run(run_path)

    # 4. COMPUTE METRICS
    # We evaluate using the most common metrics for CLEF/TREC tasks:
    # - map: Mean Average Precision (overall ranking quality)
    # - mrr@5: Mean Reciprocal Rank for the top 5 (how high is the first relevant document)
    # - ndcg@10: Normalized Discounted Cumulative Gain for the top 10 results
    print("Computing metrics...\n")
    metrics = ["mrr@5", "map", "ndcg@10", "precision@1", "recall@1000"]
    results_dict = evaluate(qrels, run, metrics, make_comparable=True)

    # 5. DISPLAY OFFICIAL RESULTS
    print("=======================================")
    print("          OFFICIAL RESULTS             ")
    print("=======================================")
    for metric, score in results_dict.items():
        # Print the score formatted to 4 decimal places
        print(f" {metric.upper():<10}: {score:.4f}")
    print("=======================================\n")


    # 6. COMPUTE METRICS BY LANGUAGE
    print("=======================================")
    print("      RESULTS BY LANGUAGE              ")
    print("=======================================")
    
    # DEFINISCI QUI LE TUE LINGUE (sigle attese nei query_id)
    languages = ["EN", "DE", "FR"] 

    # Estraiamo i dizionari di base da ranx per poterli filtrare
    qrels_dict = qrels.to_dict()
    run_dict = run.to_dict()

    for lang in languages:
        # Match the prefix the run/qrels writers actually use ("EN_", "DE_", "FR_").
        prefix = f"{lang}_"
        lang_qrels_dict = {q_id: data for q_id, data in qrels_dict.items()
                           if str(q_id).upper().startswith(prefix)}
        lang_run_dict = {q_id: data for q_id, data in run_dict.items()
                         if str(q_id).upper().startswith(prefix)}

        # Only evaluate if BOTH sides have entries AND they share at least one query.
        # Without this guard, ranx's Run() crashes with `max() arg is empty` when the
        # filtered run dict has queries that aren't in qrels (or vice versa).
        common = set(lang_qrels_dict).intersection(lang_run_dict)
        if not common:
            print(f"--- Language: {lang} ---")
            print(f" [!] No overlapping queries between run and qrels for {lang} "
                  f"(qrels: {len(lang_qrels_dict)}, run: {len(lang_run_dict)}).\n")
            continue

        # Restrict both to the common set so ranx never sees an empty inner dict.
        lang_qrels = Qrels({q: lang_qrels_dict[q] for q in common})
        lang_run = Run({q: lang_run_dict[q] for q in common})

        lang_results = evaluate(lang_qrels, lang_run, metrics, make_comparable=True)
        print(f"--- Language: {lang} ({len(common)} queries) ---")
        for metric, score in lang_results.items():
            print(f" {metric.upper():<10}: {score:.4f}")
        print()

if __name__ == "__main__":
    main()