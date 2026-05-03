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

def generate_qrels_for_split(split: str, docs_dir: str) -> str:
    """
    Build a TREC qrels file from the JSONL files of one split.
    Format per line: <query_id> 0 <doc_id> 1
    """
    output_path = os.path.join(docs_dir, f"qrels_{split}.txt")
    files = [f"en_{split}.jsonl", f"fr_{split}.jsonl", f"de_{split}.jsonl"]

    qrels_lines = []
    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            print(f"[!] Warning: {filename} not found in {docs_dir}. Skipping.")
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
                    if query_id and doc_id:
                        qrels_lines.append(f"{query_id} 0 {doc_id} 1\n")
                except json.JSONDecodeError:
                    print(f"[!] Error parsing a line in {filename}")

    if qrels_lines:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.writelines(qrels_lines)
        print(f"[SUCCESS] {output_path}: {len(qrels_lines)} judgments.\n")
    else:
        print(f"[!] No valid queries for split={split}; file not created.\n")
    return output_path


def generate_qrels():
    print("=== Starting Qrels Generation ===")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))

    # Generate qrels for BOTH splits: train (primary evaluation) and dev (final selection).
    for split in ("train", "dev"):
        generate_qrels_for_split(split, DOCS_DIR)


if __name__ == "__main__":
    generate_qrels()