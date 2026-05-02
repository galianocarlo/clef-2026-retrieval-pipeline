import lucene
import os
import sys
import time
import json
import faiss
import numpy as np
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

# Neural Models (AI)
from sentence_transformers import SentenceTransformer, CrossEncoder

# Custom Imports (Ensure paths are correct for your project)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.Parse.parse_documents import ParsedDocument
from src.Parse.parse_query import ClaimParser

# Initialize JVM for PyLucene
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

import torch
print(torch.cuda.is_available())

class Searcher:
    def __init__(self, analyzer, similarity, index_path: str, run_id: str, run_path: str, max_docs_retrieved: int):
        # 1. Initialize Lucene
        self.index_dir = Paths.get(index_path)
        self.fs_dir = FSDirectory.open(self.index_dir)
        self.reader = DirectoryReader.open(self.fs_dir)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(similarity)
        self.analyzer = analyzer

        fields_to_search = [
            "title_en", "abstract_en", 
            "title_de", "abstract_de", 
            "title_fr", "abstract_fr"
        ]
        self.qp = MultiFieldQueryParser(fields_to_search, analyzer)

        self.run_id = run_id
        self.max_docs = max_docs_retrieved
        
        # Prepare the output file
        os.makedirs(run_path, exist_ok=True)
        self.run_file_path = os.path.join(run_path, f"{run_id}.txt")
        with open(self.run_file_path, 'w', encoding='utf-8') as f:
            pass 

        # 2. Initialize FAISS (Vectors)
        print("[INFO] Loading FAISS vector database...")
        self.faiss_index = faiss.read_index(os.path.join(index_path, "corpus.faiss"))
        with open(os.path.join(index_path, "doc_ids.json"), 'r', encoding='utf-8') as f:
            self.faiss_doc_ids = json.load(f)
            
        # 3. Initialize AI Models
        print("[INFO] Loading neural models (Embedding + Re-ranker)...")
        self.dense_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device = 'cuda')
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512, device = 'cuda')
        print("[INFO] Hybrid Engine Ready!\n")

    def _get_doc_text(self, doc_id, lang):
        """Retrieves full title and abstract from Lucene for the Re-ranker."""
        query = TermQuery(Term("id", doc_id))
        hits = self.searcher.search(query, 1).scoreDocs
        if hits:
            doc = self.searcher.storedFields().document(hits[0].doc)
            title = doc.get(f"title_{lang}") or doc.get("title_en") or ""
            abstract = doc.get(f"abstract_{lang}") or doc.get("abstract_en") or ""
            return f"{title}. {abstract}"
        return ""

    def search_topics(self, parser, expected_topics: int):
        print(f"#### Starting processing of {expected_topics} queries ####")
        start_time = time.time()
        topics_count = 0
        stored_fields = self.searcher.storedFields()

        with open(self.run_file_path, 'a', encoding='utf-8') as run_file:
            for topic in parser:
                topics_count += 1
                bq_builder = BooleanQuery.Builder()
                lang = topic.language
                
                if lang == "en":
                    fields_to_search = ["title_en", "abstract_en", "title_other", "abstract_other"]
                else:
                    fields_to_search = [f"title_{lang}", f"abstract_{lang}", "title_en", "abstract_en", "title_other", "abstract_other"]

                qp = MultiFieldQueryParser(fields_to_search, self.analyzer)
                clean_text = topic.get_search_query()
                
                # --- PREPARAZIONE QUERY LUCENE ---
                if clean_text:
                    escaped_title = QueryParserBase.escape(clean_text)
                    title_query = QueryParserBase.parse(qp, escaped_title)
                    bq_builder.add(title_query, BooleanClause.Occur.SHOULD)

                journals_found = topic.extract_entities()
                if journals_found:
                    journals_text = " ".join(journals_found)
                    escaped_desc = QueryParserBase.escape(journals_text)
                    desc_query = QueryParserBase.parse(self.qp, escaped_desc)
                    bq_builder.add(desc_query, BooleanClause.Occur.SHOULD)

                final_query = bq_builder.build()

                # ====================================================
                # LO SWITCH ASIMMETRICO (IL "ROUTER")
                # ====================================================
                if lang == "en":
                    # Chiediamo direttamente a PyLucene solo i Top 5 definitivi
                    top_docs = self.searcher.search(final_query, self.max_docs) 
                    
                    results_written = 0

                    for rank, sd in enumerate(top_docs.scoreDocs):
                        doc_id = stored_fields.document(sd.doc).get(ParsedDocument.FIELDS.ID)
                        score = sd.score
                        # Scriviamo direttamente nel file TREC usando lo score di BM25
                        line = f"{lang.upper()}_{topic.query_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{self.run_id}\n"
                        run_file.write(line)

                        results_written += 1
                        
                    while results_written < self.max_docs:
                        # TAPPABUCHI per evaluation CLEF
                        fallback_doc_id = self.faiss_doc_ids[results_written] 
                        
                        line = f"{lang.upper()}_{topic.query_id}\tQ0\t{fallback_doc_id}\t{results_written}\t0.000001\t{self.run_id}\n"
                        run_file.write(line)
                        results_written += 1
                        
                        
                else:
                    # IBRIDO + NEURALE (Tedesco e Francese)
                    # Chiediamo a PyLucene 30 documenti da passare poi all'AI
                    top_docs = self.searcher.search(final_query, 30) 
                    lucene_ids = [stored_fields.document(sd.doc).get(ParsedDocument.FIELDS.ID) for sd in top_docs.scoreDocs]

                    # --- B. FAISS SEARCH (Vector) ---
                    q_vec = self.dense_model.encode([clean_text], show_progress_bar=False)
                    q_vec = np.array(q_vec, dtype=np.float32)
                    faiss.normalize_L2(q_vec)
                    _, indices = self.faiss_index.search(q_vec, 100) 
                    faiss_ids = [self.faiss_doc_ids[idx] for idx in indices[0] if idx < len(self.faiss_doc_ids)]

                    # --- C. FUSIONE E PREPARAZIONE ---
                    candidate_ids = list(set(lucene_ids + faiss_ids)) # Rimuove i duplicati
                    pairs = []
                    valid_ids = []
                    
                    chunk_size = 250 # how many words the model read simultaneously
                    overlap = 50     # blocks overlapping avoiding bad-cutting sentences

                    for d_id in candidate_ids:
                        d_text = self._get_doc_text(d_id, lang)
                        if not d_text.strip():
                            continue
                        
                        words = d_text.strip().split()

                        # if the document does not need a 'cut'
                        if len(words) <= chunk_size:
                            pairs.append([clean_text, d_text])
                            valid_ids.append(d_id)
                        else:
                            # implementing chunking
                            for i in range(0, len(words), chunk_size - overlap):
                                chunk_words = words[i:i + chunk_size]
                                chunk_text = " ".join(chunk_words)
                                pairs.append([clean_text, chunk_text])
                                valid_ids.append(d_id)

                    # --- D. NEURAL RE-RANKING ---
                    if pairs:
                        # L'AI valuta semanticamente i documenti di DE e FR
                        scores = self.reranker.predict(pairs, batch_size = 64)
                        
                        doc_max_scores = {}
                        for doc_id, score in zip(valid_ids, scores):
                            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id]:
                                doc_max_scores[doc_id] = float(score)

                        # Ordina dal punteggio più alto al più basso
                        ranked_results = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
                        
                        # Salva i Top 5 nel file TREC
                        for rank, (doc_id, score) in enumerate(ranked_results[:self.max_docs]):
                            line = f"{lang.upper()}_{topic.query_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{self.run_id}\n"
                            run_file.write(line)

                # Small visual feedback
                if topics_count % 10 == 0:
                    print(f" -> {topics_count}/{expected_topics} queries completed...")

        elapsed = int(time.time() - start_time)
        print(f"Batch completed in {elapsed} seconds.\n")

    def close(self):
        self.reader.close()


if __name__ == "__main__":
    # 1. Path Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "index"))
    RUNS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "runs"))
    DATA_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))
    JOURNALS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Parse", "journals.txt"))

    # 2. Lucene Analyzer Configuration
    similarity = BM25Similarity(0.9, 0.8)
    analyzer_map = HashMap()
    
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_en", EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_en", EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_fr", FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_fr", FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_de", GermanAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_de", GermanAnalyzer())

    multi_analyzer = PerFieldAnalyzerWrapper(StandardAnalyzer(), analyzer_map)

    # 3. Engine Initialization
    engine = Searcher(
        analyzer=multi_analyzer,
        similarity=similarity,
        index_path=INDEX_PATH,
        run_id="run-hybrid-reranked",
        run_path=RUNS_PATH,
        max_docs_retrieved=5 # CheckThat! final task usually evaluates @5
    )

    claim_parser = ClaimParser(data_dir=DATA_FOLDER, journals_file=JOURNALS_PATH)
    
    # 4. TASK DEFINITION (Validation Set Only)
    tasks = [
        {"file": "en_dev.jsonl", "lang": "en", "expected": 3905},
        {"file": "de_dev.jsonl", "lang": "de", "expected": 386},
        {"file": "fr_dev.jsonl", "lang": "fr", "expected": 702}
    ]

    try:
        # Execution
        for task in tasks:
            print(f"Processing file: {task['file']}")
            queries_iterator = claim_parser.parse_language_file(task["file"], task["lang"])
            engine.search_topics(queries_iterator, task["expected"])
    finally:
        engine.close()
        print(f"#### Hybrid Search Completed! Results saved in: {RUNS_PATH}/run-hybrid-reranked.txt ####")