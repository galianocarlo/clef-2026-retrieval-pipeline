import os
import subprocess
import sys

# 1. Dictionary of run descriptions (matching the README)
DESCRIPTIONS = {
    "seupd-baseline-run.txt": "Baseline run (initial reference provided by CLEF).",
    "run-native-analyzers.txt": "V1: Base asymmetric search using PyLucene, BM25, and native multilingual analyzers.",
    "run-native-analyzers-prf.txt": "V2: BM25 enhanced with Pseudo-Relevance Feedback (PRF) for medical term expansion.",
    "run-neural-reranked.txt": "V1 + Neural: Semantic re-ranking of the V1 Top 100 using a HuggingFace Cross-Encoder.",
    "run-neural-reranked-prf.txt": "V2 + Neural: Semantic re-ranking applied to the PRF-expanded run (our 'trawl net' approach).",
    "run-hybrid-best.txt": "V3 (Final): Hybrid pipeline combining BM25 lexical precision for English with neural re-ranking for DE and FR.",
    "run-hybrid-best-prf.txt": "V3 + PRF (Experimental): Hybrid pipeline built on the PRF expansion, combining BM25+PRF for English with Neural+PRF re-ranking for DE and FR."
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
    
    with open(report_output_path, "w", encoding="utf-8") as report_file:
        report_file.write("=========================================================\n")
        report_file.write("   SEUPD 25/26 - GLOBAL RUN METRICS REPORT               \n")
        report_file.write("=========================================================\n\n")

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

            # 4. Format and write the output to the master report
            description = DESCRIPTIONS.get(filename, "Description not available (experimental or custom run).")
            
            report_file.write(f"📁 RUN FILE: {filename}\n")
            report_file.write(f"📝 DESCRIPTION: {description}\n")
            report_file.write("-" * 40 + "\n")
            report_file.write(result.stdout) # Pastes global and per-language metrics from ranx
            report_file.write("\n\n" + ("*" * 60) + "\n\n")

    print(f"\n All metrics have been successfully saved to:\n{report_output_path}")

if __name__ == "__main__":
    main()