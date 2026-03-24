import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

DATASET_ID = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"
OUTPUT_DIR = "experiment/docs"
LANGUAGES  = ["en", "de", "fr"]

# extrac from hugging face
def extract(language: str) -> dict:
    """Download the raw dataset for a given language from HuggingFace."""
    print(f"  [EXTRACT] Downloading '{language}' dataset...")
    data = load_dataset(DATASET_ID, name=language, trust_remote_code=True)
    print(f"    -> train: {len(data['train'])} rows")
    print(f"    -> dev:   {len(data['dev'])} rows")
    return {"train": data["train"], "dev": data["dev"]}

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

def load_to_jsonl(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to JSON Lines (.jsonl) format."""
    path = os.path.join(OUTPUT_DIR, filename)
    
    df.to_json(path, orient="records", lines=True, force_ascii=False) # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
    print(f"  [LOAD] Saved {len(df)} rows -> {path}")

# VALIDATION

def validate(output_dir: str) -> None:
    """Basic validation: reload all JSONL files and print a summary."""
    print("\n  [VALIDATE] Reloading and checking output files...")

    
    expected_files = {
        "en_train.jsonl": ["index", "text", "pubkey"],
        "en_dev.jsonl":   ["index", "text", "pubkey"],
        "de_train.jsonl": ["index", "text", "pubkey"],
        "de_dev.jsonl":   ["index", "text", "pubkey"],
        "fr_train.jsonl": ["index", "text", "pubkey"],
        "fr_dev.jsonl":   ["index", "text", "pubkey"],
        "collection.jsonl": ["pubkey", "title", "venue", "authors", "abstract"],
    }

    all_ok = True
    for filename, expected_cols in expected_files.items():
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            print(f"    MISSING: {filename}")
            all_ok = False
            continue


        df = pd.read_json(path, orient="records", lines=True)

        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            print(f"    MISSING COLUMNS in {filename}: {missing_cols}")
            all_ok = False
            continue

        print(f"    OK  {filename:20s}  rows: {len(df):<6}  cols: {list(df.columns)}")

    if all_ok:
        print("\n  All files validated successfully.")
    else:
        print("\n  Some files have issues — check the output above.")


# MAIN
# ══════════════════════════════════════════════════════════════════════════════

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
    for lang in LANGUAGES:
        print(f"\nProcessing language: {lang.upper()}")

        raw_all = load_dataset(DATASET_ID, name=lang, trust_remote_code=True)

        for split in ["train", "dev"]:
            df = transform_queries(raw_all[split])
            load_to_jsonl(df, f"{lang}_{split}.jsonl")

    # Process collection
    print(f"\n{'─' * 40}")
    print("Processing: COLLECTION")
    print(f"{'─' * 40}")
    raw_col = extract_collection()
    
    df_col  = transform_collection(raw_col)
    load_to_jsonl(df_col, "collection.jsonl")

    # Check for be sure
    print(f"\n{'─' * 40}")
    validate(OUTPUT_DIR)
    print(f"{'─' * 40}")
    print("\nETL pipeline completed successfully.")
    print(f"Output files are in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()