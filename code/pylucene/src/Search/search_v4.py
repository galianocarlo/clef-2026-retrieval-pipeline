import lucene
import os
import sys
import time
import json
import faiss
import numpy as np
import torch
from tqdm import tqdm
from java.nio.file import Paths

# Lucene Imports
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.queryparser.classic import QueryParserBase, MultiFieldQueryParser
from org.apache.lucene.search.similarities import BM25Similarity

# Analyzer Imports
from java.util import HashMap
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.fr import FrenchAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer

# Neural Models
from sentence_transformers import SentenceTransformer, CrossEncoder

# Custom Imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.Parse.parse_documents import ParsedDocument
from src.Parse.parse_query import JsonArrayQueryParser
import argparse
# Initialize JVM
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

# ─────────────────────────────────────────────────────────────────────────
# Model constants — must match index_faiss.py exactly for the embedding side.
#
#   EMBEDDING_MODEL : encodes queries for FAISS lookup.
#   RERANKER_MODEL  : cross-encoder that re-scores the merged candidate pool.
#   RERANKER_MAXLEN : token budget per pair (bge-reranker-v2-m3 supports 1024).
#   RERANKER_BATCH  : pairs per GPU call.  At 568M FP16: 32 pairs ≈ 2 GB VRAM.
#
#   BM25_RECALL     : BM25 retrieves this many docs per query for the TAIL
#                     (positions beyond RERANK_POOL_SIZE are kept in BM25 order
#                     so every query emits exactly max_docs results → R@1000 intact).
#   BM25_TOP        : top-N from BM25 to include in the candidate pool.
#   FAISS_TOP       : top-N from FAISS to include in the candidate pool.
# ─────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "BAAI/bge-m3"
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"
RERANKER_MAXLEN  = 1024
RERANKER_BATCH   = 4    # safe for FP16 on 6 GB VRAM; drop to 16 if OOM

BM25_RECALL      = 1000  # full BM25 run for recall (tail padding)
BM25_TOP         = 30    # BM25 candidates fed into the re-rank pool
FAISS_TOP        = 100   # FAISS candidates fed into the re-rank pool


class Searcher:
    def __init__(self, analyzer, similarity, index_path: str,
                 run_id: str, run_path: str, max_docs_retrieved: int):

        # 1. Lucene
        self.index_dir = Paths.get(index_path)
        self.fs_dir = FSDirectory.open(self.index_dir)
        self.reader = DirectoryReader.open(self.fs_dir)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(similarity)
        self.analyzer = analyzer
        self.max_docs = max_docs_retrieved

        # Output file
        os.makedirs(run_path, exist_ok=True)
        self.run_file_path = os.path.join(run_path, f"{run_id}.txt")
        self.run_id = run_id
        with open(self.run_file_path, 'w', encoding='utf-8') as f:
            pass

        # 2. FAISS
        print("[INFO] Loading FAISS vector database...")
        self.faiss_index = faiss.read_index(os.path.join(index_path, "corpus.faiss"))
        with open(os.path.join(index_path, "doc_ids.json"), 'r', encoding='utf-8') as f:
            self.faiss_doc_ids = json.load(f)

        # 3. Neural models — FP16 on GPU to halve VRAM usage
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading dense encoder: {EMBEDDING_MODEL}  (device={device})")
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self.dense_model = SentenceTransformer(
            EMBEDDING_MODEL, model_kwargs=model_kwargs, device=device
        )

        print(f"[INFO] Loading cross-encoder: {RERANKER_MODEL}  (device={device})")
        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=RERANKER_MAXLEN,
            device=device,
        )
        print("[INFO] Hybrid Engine Ready!\n")

    # ------------------------------------------------------------------
    # Internal: look up the stored text for a single doc_id via Lucene.
    # Only called for FAISS-only candidates (those not already retrieved
    # by the BM25 phase).  Kept for correctness; performance cost is
    # bounded by (FAISS_TOP - overlap) ≤ FAISS_TOP lookups per query.
    # ------------------------------------------------------------------
    def _fetch_text(self, doc_id: str, lang: str) -> str:
        hits = self.searcher.search(TermQuery(Term("id", doc_id)), 1).scoreDocs
        if not hits:
            return ""
        doc = self.searcher.storedFields().document(hits[0].doc)
        title    = doc.get(f"title_{lang}")    or doc.get("title_en")    or ""
        abstract = doc.get(f"abstract_{lang}") or doc.get("abstract_en") or ""
        return f"{title}. {abstract}"

    def search_topics(self, parser, expected_topics: int):
        print(f"#### Starting processing of {expected_topics} queries ####")
        start_time   = time.time()
        topics_count = 0
        stored_fields = self.searcher.storedFields()

        with open(self.run_file_path, 'a', encoding='utf-8') as run_file:
            for topic in tqdm(parser, total=expected_topics, desc="Searching", unit="query"):
                topics_count += 1
                lang = topic.language

                # --- Build Lucene query (same as search.py) ---
                if lang == "en":
                    fields_to_search = ["title_en", "abstract_en", "title_other", "abstract_other"]
                else:
                    fields_to_search = [
                        f"title_{lang}", f"abstract_{lang}",
                        "title_en", "abstract_en",
                        "title_other", "abstract_other",
                    ]

                qp = MultiFieldQueryParser(fields_to_search, self.analyzer)
                bq_builder = BooleanQuery.Builder()

                clean_text = topic.get_search_query()
                if clean_text:
                    escaped = QueryParserBase.escape(clean_text)
                    bq_builder.add(QueryParserBase.parse(qp, escaped), BooleanClause.Occur.SHOULD)

                final_query = bq_builder.build()

                # ===========================================================
                # STAGE 1 — BM25 full recall run (BM25_RECALL docs)
                # Gives us the full ranked list for the tail (positions beyond
                # the rerank pool) so R@1000 is preserved.
                # ===========================================================
                bm25_full = self.searcher.search(final_query, BM25_RECALL)

                # Build an ordered list of (doc_id, bm25_score, text) from BM25
                bm25_rows = []
                for sd in bm25_full.scoreDocs:
                    doc = stored_fields.document(sd.doc)
                    d_id = doc.get(ParsedDocument.FIELDS.ID)
                    if lang == "en":
                        title    = doc.get("title_en")    or ""
                        abstract = doc.get("abstract_en") or ""
                    else:
                        title    = doc.get(f"title_{lang}")    or doc.get("title_en")    or ""
                        abstract = doc.get(f"abstract_{lang}") or doc.get("abstract_en") or ""
                    text = f"{title}. {abstract}"
                    bm25_rows.append((d_id, sd.score, text))

                # ===========================================================
                # EN fast path — BM25 is already very strong for English;
                # reranking 3000+ EN queries with a 568M cross-encoder would
                # take hours on CPU and marginal gain on EN (EN already 0.50+).
                # Revisit if runtime budget allows.
                # ===========================================================
                if lang == "en":
                    for rank, (d_id, score, _) in enumerate(bm25_rows[:self.max_docs], start=1):
                        run_file.write(
                            f"EN_{topic.query_id}\tQ0\t{d_id}\t{rank}\t{score:.6f}\t{self.run_id}\n"
                        )
                    continue

                # ===========================================================
                # DE / FR: HYBRID RETRIEVAL + NEURAL RERANKING
                # ===========================================================

                # Cache texts for the BM25 top candidates to avoid re-fetching
                bm25_top_ids  = [row[0] for row in bm25_rows[:BM25_TOP]]
                bm25_text_map = {row[0]: row[2] for row in bm25_rows[:BM25_TOP]}
                bm25_id_set   = {row[0] for row in bm25_rows}

                # STAGE 2 — FAISS dense search
                if clean_text:
                    q_vec = self.dense_model.encode(
                        [clean_text], show_progress_bar=False, convert_to_numpy=True
                    )
                    q_vec = np.array(q_vec, dtype=np.float32)
                    faiss.normalize_L2(q_vec)
                    _, indices = self.faiss_index.search(q_vec, FAISS_TOP)
                    faiss_ids = [
                        self.faiss_doc_ids[idx]
                        for idx in indices[0]
                        if idx < len(self.faiss_doc_ids)
                    ]
                else:
                    faiss_ids = []

                # STAGE 3 — Merge candidate pool (preserve order, deduplicate)
                seen = set()
                candidate_ids = []
                for d_id in bm25_top_ids + faiss_ids:
                    if d_id not in seen:
                        seen.add(d_id)
                        candidate_ids.append(d_id)

                # Build (query, doc) pairs for the cross-encoder.
                # Texts from BM25 cache are free; FAISS-only docs need a lookup.
                pairs     = []
                valid_ids = []

                for d_id in candidate_ids:
                    if d_id in bm25_text_map:
                        text = bm25_text_map[d_id]
                    else:
                        text = self._fetch_text(d_id, lang)

                    if not text.strip():
                        continue
                    
                    pairs.append([clean_text, text])
                    valid_ids.append(d_id)

                # STAGE 4 — Neural reranking
                if pairs:
                    scores = self.reranker.predict(pairs, batch_size=RERANKER_BATCH)
                
                    # Mappatura diretta score-documento (niente più Max-P)
                    doc_scores = {doc_id: float(score) for doc_id, score in zip(valid_ids, scores)}
                    
                    reranked = sorted(
                        doc_scores.items(), key=lambda x: x[1], reverse=True
                    )
                else:
                    reranked = []

                # STAGE 5 — Write output
                # Write reranked candidates first (positions 1..len(reranked)),
                # then pad with the BM25 tail so every query has max_docs results.
                written_ids = set()
                rank = 1
                for d_id, score in reranked:
                    if rank > self.max_docs:
                        break
                    lang_up = lang.upper()
                    run_file.write(
                        f"{lang_up}_{topic.query_id}\tQ0\t{d_id}\t{rank}\t{score:.6f}\t{self.run_id}\n"
                    )
                    written_ids.add(d_id)
                    rank += 1

                # Tail: BM25 docs not already in the reranked pool
                for d_id, bm25_score, _ in bm25_rows:
                    if rank > self.max_docs:    
                        break
                    if d_id not in written_ids:
                        lang_up = lang.upper()

                        # CORREZIONE: Creiamo uno score fittizio bassissimo e decrescente 
                        # in modo che ranx non lo porti mai in cima, rovinando l'MRR@5!
                        tail_score = -1000.0 - rank
                        
                        run_file.write(
                            f"{lang_up}_{topic.query_id}\tQ0\t{d_id}\t{rank}\t{tail_score:.6f}\t{self.run_id}\n" #al posto di tail_score prima c'era bm25_score
                        )
                        rank += 1

        elapsed = int(time.time() - start_time)
        print(f"Batch completed in {elapsed} seconds.\n")

    def close(self):
        self.reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev", help="Dataset split to evaluate")
    args = parser.parse_args()
    
    current_split = args.split

    # 1. Define all absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "index"))
    RUNS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "runs"))
    DATA_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))

    similarity = BM25Similarity(0.9, 0.8)

    # Per-field analyzers
    analyzer_map = HashMap()

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_en",       EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_en",    EnglishAnalyzer())

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_fr",       FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_fr",    FrenchAnalyzer())

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_de",       GermanAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_de",    GermanAnalyzer())
    # added
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_other",    StandardAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_other", StandardAnalyzer())

    multi_analyzer = PerFieldAnalyzerWrapper(StandardAnalyzer(), analyzer_map)

    current_run_id = f"run-V4_{current_split}"

    engine = Searcher(
        analyzer=multi_analyzer,
        similarity=similarity,
        index_path=INDEX_PATH,
        run_id=current_run_id,
        run_path=RUNS_PATH,
        max_docs_retrieved=1000  # standard: top 1000
    )

    # Define the batches of queries to process
    ALL_TASKS = {
        "train": [
            {"file": "en_train.jsonl", "lang": "en", "expected": 14977},
            {"file": "de_train.jsonl", "lang": "de", "expected": 1460},
            {"file": "fr_train.jsonl", "lang": "fr", "expected": 2807},
        ],
        "dev": [
            {"file": "en_dev.jsonl",  "lang": "en", "expected": 3905},
            {"file": "de_dev.jsonl",  "lang": "de", "expected": 386},
            {"file": "fr_dev.jsonl",  "lang": "fr", "expected": 702},
        ],
        "test": [
            {"file": "final_en_test.jsonl", "lang": "en", "expected": 6080},
            {"file": "final_de_test.jsonl", "lang": "de", "expected": 876},
            {"file": "final_fr_test.jsonl", "lang": "fr", "expected": 1220},
        ]
    }

    tasks = ALL_TASKS[current_split]
    try:
        for task in tasks:
            file_path = os.path.join(DATA_FOLDER, task["file"])
            if not os.path.exists(file_path):
                print(f"Skipping {task['file']} — file not found.")
                continue

            print(f"Processing: {task['file']}")
            with open(file_path, "r", encoding="utf-8") as f:
                parser = JsonArrayQueryParser(f, lang_code=task["lang"])
                engine.search_topics(parser, task["expected"])

    finally:
        engine.close()
        print(f"\nSearching Complete")