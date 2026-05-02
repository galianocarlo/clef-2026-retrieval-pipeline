import os
from ranx import Qrels, Run, evaluate
import sys
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
    if len(sys.argv) > 1:
        # Se riceve un file dal terminale, usa quello
        run_path = sys.argv[1] 
    else:
        # Altrimenti usa quello di default (metti qui quello che avevi prima)
        run_path = os.path.join(EXPERIMENT_DIR, "runs", "run-hybrid-reranked.txt")

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
    # - mrr@5: Mean Reciprocal Rank for the top 5 (how high is the first relevant document)
    # - ndcg@10: Normalized Discounted Cumulative Gain for the top 10 results
    print("Computing metrics...\n")
    metrics = ["mrr@5", "map", "ndcg@10", "precision@1", "recall@1000"]
    results_dict = evaluate(qrels, run, metrics, make_comparable=True)

    # Compute metrics by language
    languages = ["EN", "DE", "FR"] 
    qrels_dict = qrels.to_dict()
    run_dict = run.to_dict()

    lang_mrr_scores = []
    lang_results_dict = {}

    for lang in languages:
        lang_qrels_dict = {q_id: data for q_id, data in qrels_dict.items() if lang in str(q_id).upper()}
        lang_run_dict = {q_id: data for q_id, data in run_dict.items() if lang in str(q_id).upper()}

        if lang_qrels_dict and lang_run_dict:
            lang_qrels = Qrels(lang_qrels_dict)
            lang_run = Run(lang_run_dict)
            
            lang_results = evaluate(lang_qrels, lang_run, metrics, make_comparable=True)
            lang_results_dict[lang] = lang_results
            lang_mrr_scores.append(lang_results["mrr@5"])

    # Calculate Official CodaBench Macro-MRR (Average of the 3 languages MRR@5)
    mrr_score = sum(lang_mrr_scores) / len(lang_mrr_scores) if lang_mrr_scores else 0.0

    # 5. DISPLAY OFFICIAL RESULTS
    print("=======================================")
    print("          OFFICIAL RESULTS             ")
    print("=======================================")
    print(f" MRR@5  : {mrr_score:.4f}")

    for metric, score in results_dict.items():
        if metric.lower() != "mrr@5":
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
        # Filtriamo le query: teniamo solo quelle in cui la sigla della lingua appare nel query_id
        lang_qrels_dict = {q_id: data for q_id, data in qrels_dict.items() if lang in str(q_id).upper()}
        lang_run_dict = {q_id: data for q_id, data in run_dict.items() if lang in str(q_id).upper()}

        # Procediamo alla valutazione solo se ci sono effettivamente query per questa lingua
        if lang_qrels_dict and lang_run_dict:
            # Ricreiamo gli oggetti Qrels e Run specifici per la lingua
            lang_qrels = Qrels(lang_qrels_dict)
            lang_run = Run(lang_run_dict)
            
            lang_results = evaluate(lang_qrels, lang_run, metrics, make_comparable=True)
            
            print(f"--- Language: {lang} ---")
            for metric, score in lang_results.items():
                print(f" {metric.upper():<10}: {score:.4f}")
            print() # Riga vuota per separare
        else:
            print(f"--- Language: {lang} ---")
            print(" [!] Nessuna query trovata nel file di test per questa lingua.\n")

if __name__ == "__main__":
    main()