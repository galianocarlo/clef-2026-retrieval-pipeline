from datasets import load_dataset
import pandas as pd
import os
from huggingface_hub import login

# Authentication
login(token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))

DATASET_ID = "sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims"

# English
en_data = load_dataset(DATASET_ID, name="en", trust_remote_code=True)
en_train = en_data["train"]
en_dev   = en_data["dev"]

# German
de_data = load_dataset(DATASET_ID, name="de", trust_remote_code=True)
de_train = de_data["train"]
de_dev   = de_data["dev"]

# French
fr_data = load_dataset(DATASET_ID, name="fr", trust_remote_code=True)
fr_train = fr_data["train"]
fr_dev   = fr_data["dev"]

# Collection
collection_data = load_dataset(DATASET_ID, name="collection", trust_remote_code=True)
collection = collection_data["collection"]

# Info
print("=== English train ===")
print(f"Rows: {len(en_train)}")
print(f"Columns: {en_train.column_names}")
print(pd.DataFrame(en_train).head(3))

print("=== German train ===")
print(f"Rows: {len(de_train)}")
print(f"Columns: {de_train.column_names}")
print(pd.DataFrame(de_train).head(3))

print("=== French train ===")
print(f"Rows: {len(fr_train)}")
print(f"Columns: {fr_train.column_names}")
print(pd.DataFrame(fr_train).head(3))

print("\n=== Collection ===")
print(f"Rows: {len(collection)}")
print(f"Columns: {collection.column_names}")
print(pd.DataFrame(collection).head(3))