"""
**Goal:** Build the Lucene index from the parsed documents.

**What it must do:**

- Accept the list of parsed documents from `parse_documents.py`
- Write them to `experiment/docs/docs.jsonl`
- Build a Lucene index at `experiment/index/pyserini/` (or `pylucene/`)
- Print progress every 1000 documents
- Print total time taken

**Expected output:**

```
Indexing documents...
1000 documents indexed...
2000 documents indexed...
...
Index complete. Total: 50000 documents in 45.2s
Index saved to experiment/index/pyserini/
```

**Definition of done:** Index folder exists and `LuceneSearcher` can open it without errors.
"""