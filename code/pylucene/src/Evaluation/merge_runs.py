import os

def main():
    print("=== Starting Run Merge (Frankenstein Pipeline) ===")
    
    # 1. PATH SETUP
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")) 
    RUNS_PATH = os.path.join(BASE_DIR, "experiment", "runs")
    
    BM25_RUN_FILE = os.path.join(RUNS_PATH, "run-native-analyzers.txt")
    #BM25_RUN_FILE = os.path.join(RUNS_PATH, "run-native-analyzers-prf.txt")
    NEURAL_RUN_FILE = os.path.join(RUNS_PATH, "run-neural-reranked.txt")
    #NEURAL_RUN_FILE = os.path.join(RUNS_PATH, "run-neural-reranked-prf.txt")
    OUTPUT_RUN_FILE = os.path.join(RUNS_PATH, "run-hybrid-best.txt")
    #OUTPUT_RUN_FILE = os.path.join(RUNS_PATH, "run-hybrid-best-prf.txt")

    merged_lines = []
    
    # 2. GET ENGLISH FROM BM25
    print(" > Extracting ENGLISH results from BM25...")
    if os.path.exists(BM25_RUN_FILE):
        with open(BM25_RUN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("EN_"):
                    # Change the run name for clarity in the output file
                    parts = line.split()
                    parts[-1] = "hybrid-best"
                    merged_lines.append("\t".join(parts) + "\n")
                    
    # 3. GET GERMAN AND FRENCH FROM NEURAL MODEL
    print(" > Extracting DE and FR results from the Neural Model...")
    if os.path.exists(NEURAL_RUN_FILE):
        with open(NEURAL_RUN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("DE_") or line.startswith("FR_"):
                    parts = line.split()
                    parts[-1] = "hybrid-best"
                    merged_lines.append("\t".join(parts) + "\n")
                    
    # 4. SAVE THE COMBINED FILE
    with open(OUTPUT_RUN_FILE, 'w', encoding='utf-8') as f:
        f.writelines(merged_lines)
        
    print(f"\n=== Done! Created file: {OUTPUT_RUN_FILE} ===")
    print("Run evaluate.py on this file and enjoy your global score!")

if __name__ == "__main__":
    main()