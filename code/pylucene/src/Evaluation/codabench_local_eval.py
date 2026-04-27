import os
import pandas as pd
import ast

def load_qrels_trec(qrels_path):
    """
    Reads the qrels_dev.txt file (TREC format) and converts it into a nested dictionary:
    {'EN': {11: ['docA', 'docB'], ...}, 'DE': {...}}
    """
    qrels = {"EN": {}, "DE": {}, "FR": {}}
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                full_q_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                
                # Retain only documents with a strict positive relevance score
                if relevance > 0:
                    lang_prefix, numeric_id = full_q_id.split("_", 1)
                    lang = lang_prefix.upper()
                    numeric_id = int(numeric_id)
                    
                    if numeric_id not in qrels[lang]:
                        qrels[lang][numeric_id] = []
                    qrels[lang][numeric_id].append(doc_id)
                    
    return qrels

def evaluate_language(lang, qrels_lang, tsv_file_path):
    """
    Simulates the strict validation and scoring mechanism of the Codabench server for a given language.
    """
    print(f"\n--- Evaluating {lang} ---")
    if not os.path.exists(tsv_file_path):
        print(f"[!] Error: File {tsv_file_path} not found.")
        return -1.0

    try:
        # Load the predictions file
        df = pd.read_csv(tsv_file_path, sep='\t')
        
        # Convert the string representation of the list into an actual Python list
        df['preds'] = df['preds'].apply(ast.literal_eval)
        predictions = dict(zip(df['index'], df['preds']))
    except Exception as e:
        print(f"[!] Error parsing the TSV file: {e}")
        return -1.0

    # 1. STRICT CODABENCH VALIDATION
    missing_queries = []
    for q_id in qrels_lang.keys():
        if q_id not in predictions:
            missing_queries.append(q_id)
            
    if missing_queries:
        print(f"[!] Scoring Failed for {lang.lower()}! Make sure to include predictions for every index.")
        print(f"    Missing {len(missing_queries)} queries (e.g., {missing_queries[:5]}...)")
        return -1.0
        
    # 2. MRR@5 CALCULATION
    mrr_sum = 0.0
    for q_id, relevant_docs in qrels_lang.items():
        preds = predictions[q_id][:5] # Restrict evaluation to the top 5 documents
        
        score = 0.0
        for rank, p_doc in enumerate(preds):
            if str(p_doc) in relevant_docs:
                score = 1.0 / (rank + 1)
                break # Terminate search upon finding the first relevant document
        mrr_sum += score
        
    mrr_final = mrr_sum / len(qrels_lang)
    print(f"[OK] Validation successful. {lang} MRR@5: {mrr_final:.4f}")
    return mrr_final

def main():
    print("=======================================")
    print("   LOCAL CODABENCH EVALUATION SYSTEM   ")
    print("=======================================")
    
    # PATH CONFIGURATION (Adjust if necessary relative to your project structure)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    
    QRELS_PATH = os.path.join(BASE_DIR, "experiment", "docs", "qrels_dev.txt")
    SUBMISSION_DIR = os.path.join(BASE_DIR, "experiment", "submission")
    
    if not os.path.exists(QRELS_PATH):
        print(f"[!] ERROR: Unable to locate ground truth qrels at {QRELS_PATH}")
        return
        
    print("1. Loading Ground Truth (Qrels)...")
    qrels_all = load_qrels_trec(QRELS_PATH)
    
    print("2. Initiating local evaluation on TSV files...")
    results = {}
    
    for lang in ["EN", "DE", "FR"]:
        tsv_path = os.path.join(SUBMISSION_DIR, f"predictions_{lang.lower()}.tsv")
        # Restrict the evaluation keys to those present in the qrels for the current language
        mrr = evaluate_language(lang, qrels_all[lang], tsv_path)
        results[lang] = mrr
        
    print("\n=======================================")
    print("           OFFICIAL RESULTS            ")
    print("=======================================")
    
    # If any language evaluation fails (-1), the global score equates to -1 (mirroring Codabench behavior)
    if any(score == -1 for score in results.values()):
        print("[!] GLOBAL EVALUATION FAILED.")
        print("Please resolve the errors detailed above prior to submitting to Codabench.")
    else:   
        macro_mrr = sum(results.values()) / 3.0
        print(f"CLEF OFFICIAL MACRO-MRR@5 : {macro_mrr:.4f}")
        print("This metric reflects the expected score on the official leaderboard.")

if __name__ == "__main__":
    main()