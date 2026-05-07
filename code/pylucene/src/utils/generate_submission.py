import os
import sys
import zipfile
from collections import defaultdict

# =============================================================================
# SUBMISSION GENERATOR — Codabench TSV Format
# -----------------------------------------------------------------------------
# Reads a TREC run file (output of search.py) and converts it to the three
# TSV files expected by Codabench:
#   predictions_en.tsv, predictions_de.tsv, predictions_fr.tsv
# Then zips them into predictions.zip ready for upload.
#
# Usage:
#   python generate_submission.py                          (uses default run)
#   python generate_submission.py path/to/run-V*_{split}.txt
# =============================================================================

LANGUAGES = ["EN", "DE", "FR"]
TOP_K = 5  # Competition requires exactly top-5

def parse_trec_run(run_path):
    """
    Reads a TREC run file and returns a nested dict:
      { "EN": { "1": ["docA", "docB", ...], "2": [...] }, "DE": {...}, "FR": {...} }
    Results are already ordered by rank (TREC format guarantees this).
    """
    results = {lang: defaultdict(list) for lang in LANGUAGES}

    with open(run_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue  # skip malformed lines

            full_query_id = parts[0]          # e.g. "EN_1"
            doc_id        = parts[2]          # e.g. "2304.05602"
            # parts[1] = Q0, parts[3] = rank, parts[4] = score

            # Split "EN_1" → lang="EN", query_num="1"
            if '_' not in full_query_id:
                continue
            lang, query_num = full_query_id.split('_', 1)
            lang = lang.upper()

            if lang not in results:
                continue

            results[lang][query_num].append(doc_id)

    return results

def write_tsv(results_for_lang, output_path):
    """
    Writes one predictions_XX.tsv file.
    Format:
        index\tpreds
        1\t['docA', 'docB', 'docC', 'docD', 'docE']
    """
    # Sort queries numerically
    sorted_queries = sorted(results_for_lang.items(), key=lambda x: int(x[0]))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("index\tpreds\n")
        for query_num, doc_ids in sorted_queries:
            top5 = doc_ids[:TOP_K]

            # Pad with 0 if fewer than 5 results (safety net — use int like the notebook)
            while len(top5) < TOP_K:
                top5.append("0")

            # Format as unquoted integers, matching the notebook's pandas/numpy output:
            # [2764, 7487, 5334, 5968, 2627]  ← no quotes around the numbers
            preds_str = "[" + ", ".join(d for d in top5) + "]"
            f.write(f"{query_num}\t{preds_str}\n")

    print(f"  Written {len(sorted_queries)} queries → {output_path}")

def main():
    SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment"))
    RUNS_DIR      = os.path.join(EXPERIMENT_DIR, "runs")
    SUBMIT_DIR    = os.path.join(EXPERIMENT_DIR, "submission")
    os.makedirs(SUBMIT_DIR, exist_ok=True)

    # Accept run file from command line or fall back to best known run
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
    else:
        run_path = os.path.join(RUNS_DIR, "run-V4_test.txt")

    if not os.path.exists(run_path):
        print(f"[!] Run file not found: {run_path}")
        return

    print(f"Reading run: {run_path}")
    results = parse_trec_run(run_path)

    # Write one TSV per language
    tsv_files = []
    for lang in LANGUAGES:
        if not results[lang]:
            print(f"No results found for language {lang} — skipping")
            continue
        tsv_path = os.path.join(SUBMIT_DIR, f"predictions_{lang.lower()}.tsv")
        write_tsv(results[lang], tsv_path)
        tsv_files.append(tsv_path)

    # Zip all three files into predictions.zip
    zip_path = os.path.join(SUBMIT_DIR, "predictions.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for tsv_path in tsv_files:
            zf.write(tsv_path, arcname=os.path.basename(tsv_path))

    print(f"\nSubmission ready: {zip_path}")
    print(f"   Contains: {[os.path.basename(f) for f in tsv_files]}")

if __name__ == "__main__":
    main()