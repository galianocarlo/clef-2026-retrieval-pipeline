import os
import json

# =============================================================================
# QRELS GENERATOR 
# -----------------------------------------------------------------------------
# This script extracts the ground truth (Query ID and target Document ID) 
# from the CLEF development JSONL files and formats them into a standard 
# TREC qrels text file.
# Format: <query_id> 0 <doc_id> 1
# =============================================================================

def generate_qrels():
    print("=== Starting Qrels Generation ===")
    
    # 1. PATH RESOLUTION
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Navigate UP three levels to reach the 'code' folder, then DOWN to 'experiment/docs'
    DOCS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))
    
    # Output file path
    output_qrels_path = os.path.join(DOCS_DIR, "qrels_dev.txt")
    
    # The files we want to extract answers from
    dev_files = ["en_dev.jsonl", "fr_dev.jsonl", "de_dev.jsonl"]
    
    qrels_lines = []
    
    # 2. PROCESS FILES
    for filename in dev_files:
        filepath = os.path.join(DOCS_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"[!] Warning: {filename} not found in {DOCS_DIR}. Skipping.")
            continue
            
        print(f"Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                
                try:
                    entry = json.loads(line)
                    lang_prefix = filename[:2].upper()
                    query_id = f"{lang_prefix}_{str(entry.get('index', ''))}"
                    doc_id = str(entry.get("pubkey", ""))
                    
                    # If both fields exist, format them according to TREC standard
                    if query_id and doc_id:
                        # Format: Query_ID  Iteration(always 0)  Doc_ID  Relevance(always 1 for positive match)
                        qrels_lines.append(f"{query_id} 0 {doc_id} 1\n")
                except json.JSONDecodeError:
                    print(f"[!] Error parsing a line in {filename}")

    # 3. SAVE RESULTS
    if qrels_lines:
        with open(output_qrels_path, 'w', encoding='utf-8') as out_file:
            out_file.writelines(qrels_lines)
        print(f"\n[SUCCESS] Generated {output_qrels_path} with {len(qrels_lines)} relevance judgments.")
    else:
        print("\n[!] No valid queries found. Qrels file was not created.")

if __name__ == "__main__":
    generate_qrels()