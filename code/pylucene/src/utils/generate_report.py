import os
import subprocess
import sys
import re

# 1. Dictionary of run prefixes
DESCRIPTIONS = {
    "run-v1": "V1: Lexical Baseline (BM25, Native Analyzers)",
    "run-v2": "V2: Statistical PRF (Pseudo-Relevance Feedback)",
    "run-V3": "V3: Hybrid Asymmetric Search (e5-base + Cross-Encoder + Chunking)",
    "run-V4": "V4: SOTA Hybrid Architecture (bge-m3 + Cross-Encoder, No Chunking)"
}

def get_description(filename):
    for key, desc in DESCRIPTIONS.items():
        if filename.startswith(key):
            return desc
    return "Description not available."

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RUNS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "experiment", "runs"))
    RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "utils"))
    EVAL_SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "Evaluation", "evaluate.py"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_output_path = os.path.join(RESULTS_DIR, "master_metrics_report.txt")

    print(f"🔍 Scanning directory: {RUNS_DIR}")
    
    dev_results = []
    train_results = []

    for filename in sorted(os.listdir(RUNS_DIR)):
        file_path = os.path.join(RUNS_DIR, filename)
        
        if not filename.endswith(".txt") or os.path.getsize(file_path) == 0:
            continue
            
        if "test" in filename.lower():
            print(f"Skipping: {filename} (Test set lacks ground truth).")
            continue
            
        split_arg = "train" if "train" in filename.lower() else "dev"
        print(f"Evaluating: {filename} on {split_arg.upper()} set...")
        
        # DIFESA 1 e 2: sys.executable assicura l'ambiente corretto, encoding previene crash Windows
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT, file_path, "--split", split_arg], 
            capture_output=True, 
            text=True,
            encoding="utf-8"
        )
        
        # DIFESA 3: Se evaluate.py fallisce brutalmente, stampiamo l'errore REALE di Python
        if result.returncode != 0:
            print(f"CRASH evaluating {filename}. Skipping.")
            print(f"   --> error: {result.stderr.strip()}")
            continue

        # Se evaluate.py è uscito correttamente ma non ha trovato i file qrels
        if "[!] Error:" in result.stdout:
            print(f"FILE MANCANTE in {filename}:")
            print(f"   --> {result.stdout.strip()}")
            continue

        mrr_match = re.search(r"MACRO MRR@5\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        r1000_match = re.search(r"MACRO RECALL@1000\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        p1_match = re.search(r"MACRO PRECISION@1\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)

        if not mrr_match: 
            mrr_match = re.search(r"MRR@5\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        if not r1000_match: 
            r1000_match = re.search(r"RECALL@1000\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)
        if not p1_match: 
            p1_match = re.search(r"(?:P|PRECISION)@1\s*:\s*([0-9.]+)", result.stdout, re.IGNORECASE)

        res_dict = {
            "filename": filename,
            "description": get_description(filename),
            "mrr": float(mrr_match.group(1)) if mrr_match else 0.0,
            "p1": float(p1_match.group(1)) if p1_match else 0.0,
            "r1000": float(r1000_match.group(1)) if r1000_match else 0.0,
            "full_output": result.stdout
        }

        if split_arg == "train":
            train_results.append(res_dict)
        else:
            dev_results.append(res_dict)
    
    dev_results.sort(key=lambda x: x["mrr"], reverse=True)
    train_results.sort(key=lambda x: x["mrr"], reverse=True)

    print("\n📝 Stesura del report finale in corso...")
    with open(report_output_path, "w", encoding="utf-8") as report_file:
        
        report_file.write("=========================================================\n")
        report_file.write("   SEUPD 25/26 - GLOBAL RUN METRICS REPORT               \n")
        report_file.write("=========================================================\n\n")
        
        report_file.write("🏆 LEADERBOARD - DEV SET (Ordinata per MACRO MRR@5)\n")
        report_file.write("-" * 135 + "\n")
        report_file.write(f"| {'Pos':<3} | {'Run File':<25} | {'MRR@5':<7} | {'P@1':<7} | {'R@1000':<7} | {'Descrizione'}\n")
        report_file.write("-" * 135 + "\n")
        for i, res in enumerate(dev_results, 1):
            report_file.write(f"| {i:<3} | {res['filename']:<25} | {res['mrr']:.4f}  | {res['p1']:.4f}  | {res['r1000']:.4f}   | {res['description']}\n")
        report_file.write("-" * 135 + "\n\n")

        report_file.write("🏆 LEADERBOARD - TRAIN SET (Ordinata per MACRO MRR@5)\n")
        report_file.write("-" * 135 + "\n")
        report_file.write(f"| {'Pos':<3} | {'Run File':<25} | {'MRR@5':<7} | {'P@1':<7} | {'R@1000':<7} | {'Descrizione'}\n")
        report_file.write("-" * 135 + "\n")
        for i, res in enumerate(train_results, 1):
            report_file.write(f"| {i:<3} | {res['filename']:<25} | {res['mrr']:.4f}  | {res['p1']:.4f}  | {res['r1000']:.4f}   | {res['description']}\n")
        report_file.write("-" * 135 + "\n\n")

        report_file.write("📄 RUNS DETAILS\n")
        report_file.write("=========================================================\n\n")
        
        for res_list in [dev_results, train_results]:
            for i, res in enumerate(res_list, 1):
                report_file.write(f"RUN FILE: {res['filename']}\n")
                report_file.write(f"DESCRIPTION: {res['description']}\n")
                report_file.write("-" * 40 + "\n")
                report_file.write(res['full_output'])
                report_file.write("\n" + ("*" * 60) + "\n\n")

    print(f"\n All metrics have been successfully saved to:\n{report_output_path}")

if __name__ == "__main__":
    main()