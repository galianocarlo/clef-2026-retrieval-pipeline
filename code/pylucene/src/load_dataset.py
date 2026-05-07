import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download

DATASET_ID = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
# 1. Get the absolute path of the directory where this script lives
# This will be: C:/.../pylucene/src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up TWO levels to reach the project root, then into experiment/docs
# Path: src -> pylucene -> (Root) -> experiment -> docs
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))
print(f"Targeting directory: {OUTPUT_DIR}")
LANGUAGES  = ["en", "de", "fr"]

# Around line 12:
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "experiment", "docs"))
CORPUS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "experiment", "corpus")) # NEW!

# extrac from hugging face
def extract(language: str) -> dict:
    """Download the raw dataset for a given language from HuggingFace."""
    print(f"  [EXTRACT] Downloading '{language}' dataset...")
    data = load_dataset(DATASET_ID, name=language, trust_remote_code=True)
    print(f"    -> train: {len(data.get('train', []))} rows")
    print(f"    -> dev:   {len(data.get('dev', []))} rows")
    print(f"    -> test:  {len(data.get('test', []))} rows")
    return {"train": data.get("train"), "dev": data.get("dev"), "test": data.get("test")}

def extract_collection() -> object:
    """Download the raw paper collection from HuggingFace."""
    print("  [EXTRACT] Downloading paper collection...")
    data = load_dataset(DATASET_ID, name="collection", trust_remote_code=True)
    collection = data["collection"]
    print(f"    -> collection: {len(collection)} rows")
    return collection

# TRANSFORM (to do)

def transform_queries(raw_dataset) -> pd.DataFrame:
    """Transform a raw query dataset into a clean normalized DataFrame."""
    df = pd.DataFrame(raw_dataset)
    return df

def transform_collection(raw_dataset) -> pd.DataFrame:
    """Transform the raw paper collection into a clean normalized DataFrame."""
    df = pd.DataFrame(raw_dataset)
    return df

# LOAD

def load_to_jsonl(df: pd.DataFrame, filename: str, target_dir: str = OUTPUT_DIR) -> None:
    """Save a DataFrame to JSON Lines (.jsonl) format."""
    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    path = os.path.join(target_dir, filename)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"  [LOAD] Saved {len(df)} rows -> {path}")

# VALIDATION

def validate(output_dir: str) -> None:
    """Basic validation: reload all JSONL files, check schemas, and verify row counts."""
    print("\n  [VALIDATE] Reloading and checking output files...")

    # We map the filename to its expected columns, its specific folder, and an optional row count
    expected_files = {
        "en_train.jsonl":         {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "en_dev.jsonl":           {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "final_en_test.jsonl":    {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "de_train.jsonl":         {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "de_dev.jsonl":           {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "final_de_test.jsonl":    {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "fr_train.jsonl":         {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "fr_dev.jsonl":           {"cols": ["index", "text", "pubkey"], "dir": output_dir},
        "final_fr_test.jsonl":    {"cols": ["index", "text", "pubkey"], "dir": output_dir},

        "collection.jsonl": {"cols": ["pubkey", "title", "venue", "authors", "abstract"], 
                             "dir": output_dir, 
                             "expected_rows": 10000}, # https://huggingface.co/datasets/sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims/viewer/collection
    }

    all_ok = True
    for filename, rules in expected_files.items():
        path = os.path.join(rules["dir"], filename)
        
        if not os.path.exists(path):
            print(f"MISSING FILE: {path}")
            all_ok = False
            continue

        # Load the JSONL to verify it isn't corrupted
        df = pd.read_json(path, orient="records", lines=True)

        # 1. Check Columns
        missing_cols = [c for c in rules["cols"] if c not in df.columns]
        if missing_cols:
            print(f"MISSING COLUMNS in {filename}: {missing_cols}")
            all_ok = False
            continue
            
        # 2. Check Row Counts (if we provided an expected_rows rule)
        actual_rows = len(df)
        expected_rows = rules.get("expected_rows")
        
        if expected_rows and actual_rows != expected_rows:
            print(f"ROW MISMATCH in {filename}: Expected {expected_rows}, got {actual_rows}")
            all_ok = False
            continue

        print(f"{filename:20s}  rows: {actual_rows:<6}  cols: {list(df.columns)}")

    if all_ok:
        print("\n  All files validated successfully. Data is pristine.")
    else:
        print("\n  Some files have issues — check the output above.")


# MAIN
def main():

    # Authenticate
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "HUGGING_FACE_HUB_TOKEN not found. "
            "Make sure your .env file exists and docker-compose.yml has env_file: - .env"
        )
    login(token=token)
    print("HuggingFace authentication successful.\n")

    # Create output directory 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each language
    # Process each language
    for lang in LANGUAGES:
        print(f"\nProcessing language: {lang.upper()}")

        # 1. Download Train e Dev sets
        raw_all = load_dataset(DATASET_ID, name=lang, trust_remote_code=True)
        for split in ["train", "dev"]:
            df = transform_queries(raw_all[split])
            load_to_jsonl(df, f"{lang}_{split}.jsonl")

        # 2. Download Test set "manually"
        try:
            test_file = hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=f"final_{lang}_test.json")
            load_to_jsonl(pd.read_json(test_file), f"final_{lang}_test.jsonl")
        except Exception as e:
            print(f"File di test non trovato per {lang}: {e}")

    # Process collection
    print(f"\n{'─' * 40}")
    print("Processing: COLLECTION")
    print(f"{'─' * 40}")
    raw_col = extract_collection()
    
    df_col  = transform_collection(raw_col)
    load_to_jsonl(df_col, "collection.jsonl", target_dir=OUTPUT_DIR)

    # Check for be sure
    print(f"\n{'─' * 40}")
    validate(OUTPUT_DIR)
    print(f"{'─' * 40}")
    print("\nETL pipeline completed successfully.")
    print(f"Output files are in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()