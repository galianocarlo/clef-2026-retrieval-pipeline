import os
import subprocess
import sys
import re

# 1. Dictionary of run descriptions (matching the README)
DESCRIPTIONS = {
    "seupd-baseline-run.txt": "Baseline run (initial reference provided by CLEF).",
    "run-native-analyzers.txt": "V1: Base asymmetric search using PyLucene, BM25, and native multilingual analyzers.",
    "run-native-analyzers-prf.txt": "V2: BM25 enhanced with Pseudo-Relevance Feedback (PRF) for medical term expansion.",
    "run-neural-reranked.txt": "V1 + Neural: Semantic re-ranking of the V1 Top 100 using a HuggingFace Cross-Encoder.",
    "run-neural-reranked-prf.txt": "V2 + Neural: Semantic re-ranking applied to the PRF-expanded run.",
    "run-hybrid-best.txt": "V3 (Final): Hybrid pipeline combining BM25 lexical precision for English with neural re-ranking for DE and FR.",
    "run-hybrid-best-prf.txt": "V3 + PRF (Experimental): Hybrid pipeline built on the PRF expansion, combining BM25+PRF for English with Neural+PRF re-ranking for DE and FR.",
    "run-hybrid-reranked.txt": "V4 (Asymmetric Hybrid): Language-routed pipeline using pure lexical BM25 for English queries, and a Dense+Lexical Hybrid approach with Neural Re-ranking strictly for German and French."
}

def main():
    # 2. Relative paths setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path navigation: Evaluation -> src -> pylucene -> code -> experiment -> runs
    RUNS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "experiment", "runs"))
    
    # Path navigation: Evaluation -> src -> pylucene -> code -> root -> results
    RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Results"))
    
    EVAL_SCRIPT = os.path.join(BASE_DIR, "evaluate.py")

    # Ensure the results directory exists in the root
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    report_output_path = os.path.join(RESULTS_DIR, "master_metrics_report.txt")

    print(f"🔍 Scanning directory: {RUNS_DIR}")
    
    # List that will contain the dictionaries with all the scores
    all_results = []

    # 3. Iterate over all files in the runs directory
    for filename in sorted(os.listdir(RUNS_DIR)):
        file_path = os.path.join(RUNS_DIR, filename)
        
        # Skip .gitkeep, non-txt files, or empty files
        if not filename.endswith(".txt") or os.path.getsize(file_path) == 0:
            continue
        
        print(f"Evaluating: {filename}...")
        
        # Execute evaluate.py passing the specific file and capture terminal output
        result = subprocess.run(
            ["python", EVAL_SCRIPT, file_path], 
            capture_output=True, 
            text=True
        )
        
        # Skip if evaluation fails
        if result.returncode != 0:
            print(f" Error evaluating {filename}. Skipping.")
            continue

        # Extract the metrics with regex
        mrr_match = re.search(r"MRR@5\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        map_match = re.search(r"MAP\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        ndcg_match = re.search(r"NDCG@10\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        p1_match = re.search(r"(?:P|PRECISION)@1\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        r1000_match = re.search(r"RECALL@1000\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)

        mrr_score = float(mrr_match.group(1)) if mrr_match else 0.0
        map_score = float(map_match.group(1)) if map_match else 0.0
        ndcg_score = float(ndcg_match.group(1)) if ndcg_match else 0.0
        p1_score = float(p1_match.group(1)) if p1_match else 0.0
        r1000_score = float(r1000_match.group(1)) if r1000_match else 0.0

        all_results.append({
            "filename": filename,
            "description": DESCRIPTIONS.get(filename, "Description not available."),
            "mrr": mrr_score,
            "map": map_score,
            "ndcg": ndcg_score,
            "p1": p1_score,
            "r1000": r1000_score,    
            "full_output": result.stdout
        })
    
    all_results.sort(key=lambda x: x["mrr"], reverse=True)

    print("\n📝 Stesura del report finale in corso...")
    with open(report_output_path, "w", encoding="utf-8") as report_file:
        
        # --- SEZIONE 1: LEADERBOARD ---
        report_file.write("=========================================================\n")
        report_file.write("   SEUPD 25/26 - GLOBAL RUN METRICS REPORT               \n")
        report_file.write("=========================================================\n\n")
        
        report_file.write("🏆 LEADERBOARD (Ordinata per MRR@5 Globale)\n")
        report_file.write("-" * 125 + "\n")
        report_file.write(f"| {'Pos':<3} | {'Run File':<32} | {'MRR@5':<7} | {'P@1':<7} | {'R@1000':<7} | {'Descrizione'}\n")
        report_file.write("-" * 125 + "\n")
        
        for i, res in enumerate(all_results, 1):
            report_file.write(f"| {i:<3} | {res['filename']:<32} | {res['mrr']:.4f}  | {res['p1']:.4f}  | {res['r1000']:.4f}   | {res['description']}\n")
        
        report_file.write("-" * 125 + "\n\n")
        
        # --- SEZIONE 2: DETTAGLI DELLE RUN ---
        report_file.write("📄 DETTAGLI PER SINGOLA RUN (Ordinate dalla migliore alla peggiore)\n")
        report_file.write("=========================================================\n\n")
        
        for i, res in enumerate(all_results, 1):
            report_file.write(f"🏆 Posizione in classifica: {i}\n")
            report_file.write(f"📁 RUN FILE: {res['filename']}\n")
            report_file.write(f"📝 DESCRIPTION: {res['description']}\n")
            report_file.write("-" * 40 + "\n")
            report_file.write(res['full_output'])
            report_file.write("\n" + ("*" * 60) + "\n\n")
    print(f"\n All metrics have been successfully saved to:\n{report_output_path}")

if __name__ == "__main__":
    main()