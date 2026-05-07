import os
import argparse
from tqdm import tqdm
from ranx import Qrels, Run, evaluate

# =============================================================================
# SEARCH ENGINE EVALUATOR (TREC Format - Codabench Macro-Average)
# =============================================================================

def _count_lines(path: str) -> int:
    """Count non-empty lines so tqdm has a known total."""
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())

def load_trec_run(path: str) -> Run:
    total = _count_lines(path)
    run_dict: dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Loading run", unit="line"):
            parts = line.split()
            if len(parts) < 6: continue
            q_id, _, doc_id, _, score, _ = parts[:6]
            run_dict.setdefault(q_id, {})[doc_id] = float(score)
    return Run(run_dict)

def load_trec_qrels(path: str) -> Qrels:
    total = _count_lines(path)
    qrels_dict: dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Loading qrels", unit="line"):
            parts = line.split()
            if len(parts) < 4: continue
            q_id, _, doc_id, rel = parts[:4]
            qrels_dict.setdefault(q_id, {})[doc_id] = int(rel)
    return Qrels(qrels_dict)

def main():
    print("=== Starting Search Engine Evaluation ===")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment"))

    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", help="Path to the TREC run file produced by search.py")
    parser.add_argument("--split", choices=["train", "dev"], default="dev",
                        help="Which split's qrels to evaluate against")
    args = parser.parse_args()

    qrels_path = os.path.join(EXPERIMENT_DIR, "docs", f"qrels_{args.split}.txt")
    run_path = args.run_file

    if not os.path.exists(qrels_path):
        print(f"[!] Error: Qrels file not found at {qrels_path}") 
        return
    if not os.path.exists(run_path):
        print(f"[!] Error: Run file not found at {run_path}") 
        return

    print(f"\nEvaluating: {os.path.basename(run_path)} against qrels_{args.split}.txt")
    qrels = load_trec_qrels(qrels_path)
    run = load_trec_run(run_path)

    metrics = ["mrr@5", "map", "ndcg@10", "precision@1", "recall@1000"]
    languages = ["EN", "DE", "FR"] 

    qrels_dict = qrels.to_dict()
    run_dict = run.to_dict()
    
    lang_scores_accumulator = {m: [] for m in metrics}

    print("\n=======================================")
    print("      RESULTS BY LANGUAGE              ")
    print("=======================================")

    for lang in languages:
        prefix = f"{lang}_"
        lang_qrels_dict = {q: d for q, d in qrels_dict.items() if str(q).upper().startswith(prefix)}
        lang_run_dict = {q: d for q, d in run_dict.items() if str(q).upper().startswith(prefix)}

        common = set(lang_qrels_dict).intersection(lang_run_dict)
        if not common:
            print(f"--- Language: {lang} --- [!] No overlapping queries.\n")
            continue

        lang_qrels = Qrels({q: lang_qrels_dict[q] for q in common})
        lang_run = Run({q: lang_run_dict[q] for q in common})

        lang_results = evaluate(lang_qrels, lang_run, metrics, make_comparable=True)
        print(f"--- Language: {lang} ({len(common)} queries) ---")
        for metric, score in lang_results.items():
            print(f" {metric.upper():<10}: {score:.4f}")
            lang_scores_accumulator[metric].append(score)
        print()

    print("==========================")
    print("  OFFICIAL MACRO-AVERAGE  ")
    print("==========================")
    # Calcola la media aritmetica (diviso 3) esattamente come la piattaforma CLEF
    for metric in metrics:
        scores = lang_scores_accumulator[metric]
        if len(scores) == len(languages):
            macro_avg = sum(scores) / len(languages)
            print(f" MACRO {metric.upper():<10}: {macro_avg:.4f}")
        else:
            print(f" MACRO {metric.upper():<10}: N/A (Missing languages in run)")
    print("==========================\n")

if __name__ == "__main__":
    main()