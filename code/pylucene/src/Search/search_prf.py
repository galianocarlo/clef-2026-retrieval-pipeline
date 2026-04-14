import lucene
import os
import sys
import time
import re
from collections import Counter
from java.nio.file import Paths

# Core Lucene search tools
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.queryparser.classic import QueryParser, QueryParserBase, MultiFieldQueryParser

# Text Analysis and Scoring (Must match the Indexer exactly!)
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

# Import our custom parsers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.Parse.parse_documents import ParsedDocument
from src.Parse.parse_query import ClaimParser

# =============================
# == IMPORTING NEW ANALYZERS ==
# =============================
from java.util import HashMap
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.fr import FrenchAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer

# ==========================================
# --- PRF: MULTILINGUAL STOPWORDS ---
# ==========================================
# We must filter out useless words in ALL languages to avoid 
# polluting the expanded query with articles and prepositions.
MULTILINGUAL_STOPWORDS = set([
    # English
    "is", "the", "of", "and", "to", "in", "for", "with", "on", "that", "this", "it", "as", "at", "by", "from", "are", "was", "were", "be", "an",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "en", "au", "aux", "pour", "dans", "sur", "par", "qui", "que", "pas", "est", "ce", "il", "elle",
    # German
    "der", "die", "das", "und", "in", "den", "von", "zu", "dem", "für", "mit", "ist", "auf", "des", "ein", "eine", "nicht", "als", "dass", "es", "sich", "im",
    # Italian (just in case)
    "il", "lo", "la", "i", "gli", "le", "un", "una", "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "e", "o", "ma", "se", "non"
])

# Initialize PyLucene
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

class Searcher:
    """
    Searches a document collection.
    Directly replicates the Professor's Searcher.java logic in Python.
    """
    def __init__(self, analyzer, similarity, index_path: str, run_id: str, run_path: str, max_docs_retrieved: int):
        
        # 1. Open the Index
        self.index_dir = Paths.get(index_path)
        self.fs_dir = FSDirectory.open(self.index_dir)
        self.reader = DirectoryReader.open(self.fs_dir)
        
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(similarity)

        self.analyzer = analyzer
        
        # 2. Testing langdetect bugs that I (Carlo) swa in Luke
        fields_to_search = [
            "title_en", "abstract_en", 
            "title_de", "abstract_de", 
            "title_fr", "abstract_fr"
        ]
        # Implementing MultiFieldQueryParser
        self.qp = MultiFieldQueryParser(fields_to_search, analyzer)
        
        # 3. Setup the Output File
        self.run_id = run_id
        self.max_docs = max_docs_retrieved
        
        os.makedirs(run_path, exist_ok=True)
        self.run_file_path = os.path.join(run_path, f"{run_id}.txt")
        
        # We open in 'w' (write) mode to clear previous runs
        with open(self.run_file_path, 'w', encoding='utf-8') as f:
            pass 

    def search_topics(self, parser, expected_topics: int):
        """
        Executes the search for a parsed batch of topics and appends to the run file.
        """
        print(f"\n#### Start searching {expected_topics} topics ####")
        start_time = time.time()
        topics_count = 0
        
        stored_fields = self.searcher.storedFields()

        with open(self.run_file_path, 'a', encoding='utf-8') as run_file:
            
            for topic in parser:
                topics_count += 1
                bq_builder = BooleanQuery.Builder()

                # --- Building the Parser w.r.t. languages of the queries ---
                lang = topic.language
                if lang == "en":
                    fields_to_search = ["title_en", "abstract_en", "title_other", "abstract_other"]
                else:
                    fields_to_search = [f"title_{lang}", f"abstract_{lang}", "title_en", "abstract_en", "title_other", "abstract_other"]

                qp = MultiFieldQueryParser(fields_to_search, self.analyzer)
                
                # --- A. THE MAIN QUERY (The cleaned tweet) ---
                clean_text = topic.get_search_query()
                if clean_text:
                    clean_text = clean_text.replace("-", "")
                    escaped_title = QueryParserBase.escape(clean_text)
                    title_query = QueryParserBase.parse(qp, escaped_title)
                    bq_builder.add(title_query, BooleanClause.Occur.SHOULD)

                # --- B. THE BOOST QUERY (The extracted journals) ---
                journals_found = topic.extract_entities()
                if journals_found:
                    journals_text = " ".join(journals_found)
                    escaped_desc = QueryParserBase.escape(journals_text)
                    desc_query = QueryParserBase.parse(self.qp, escaped_desc)
                    bq_builder.add(desc_query, BooleanClause.Occur.SHOULD)
                
                # Finalize the initial base query
                final_query = bq_builder.build()

                # =====================================================================
                # --- PRF: TWO-STAGE RETRIEVAL IMPLEMENTATION ---
                # =====================================================================
                
                # STEP 1: Fast initial search (Top 3 documents only) to spy on their vocabulary
                initial_docs = self.searcher.search(final_query, 3)
                
                # STEP 2: Extract text from these top 3 documents
                extracted_texts = []
                for sd in initial_docs.scoreDocs:
                    doc = stored_fields.document(sd.doc)
                    
                    # 1. Always retrieve the English and 'other' fields
                    t_en = doc.get("title_en") or ""
                    a_en = doc.get("abstract_en") or ""
                    t_other = doc.get("title_other") or ""
                    a_other = doc.get("abstract_other") or ""
                    
                    # 2. DYNAMICALLY retrieve the native language fields if the tweet is not in English!
                    t_native = ""
                    a_native = ""
                    if lang != "en":
                        t_native = doc.get(f"title_{lang}") or ""
                        a_native = doc.get(f"abstract_{lang}") or ""
                        
                    # 3. Combine all retrieved texts into a single super-string
                    extracted_texts.append(f"{t_native} {a_native} {t_en} {a_en} {t_other} {a_other}")
                
                # Combine all text, make it lowercase, and extract words of at least 4 letters
                combined_text = " ".join(extracted_texts).lower()
                words = re.findall(r'\b[a-z]{4,}\b', combined_text)
                
                # Filter out stopwords and find the 5 most common terms
                useful_words = [w for w in words if w not in MULTILINGUAL_STOPWORDS]
                counts = Counter(useful_words)
                prf_terms = [word for word, freq in counts.most_common(5)]
                
                # STEP 3: Build the EXPANDED QUERY
                expanded_builder = BooleanQuery.Builder()
                
                # Add the original query requirements
                expanded_builder.add(final_query, BooleanClause.Occur.MUST)
                
                # Add the newly discovered PRF terms to boost relevance
                if prf_terms:
                    prf_text = " ".join(prf_terms)
                    escaped_prf = QueryParserBase.escape(prf_text)
                    prf_query = QueryParserBase.parse(qp, escaped_prf)
                    expanded_builder.add(prf_query, BooleanClause.Occur.SHOULD)
                    
                super_query = expanded_builder.build()

                # STEP 4: EXECUTE FINAL SEARCH with the expanded query
                top_docs = self.searcher.search(super_query, self.max_docs)
                score_docs = top_docs.scoreDocs
                
                # =====================================================================
                # --- END OF PRF ---
                # =====================================================================

                # --- E. WRITE TO TREC FORMAT ---
                for rank, sd in enumerate(score_docs):
                    doc = stored_fields.document(sd.doc)
                    doc_id = doc.get(ParsedDocument.FIELDS.ID)
                    
                    line = f"{topic.language.upper()}_{topic.query_id}\tQ0\t{doc_id}\t{rank}\t{sd.score:.6f}\t{self.run_id}\n"
                    run_file.write(line)

        elapsed = int(time.time() - start_time)
        print(f"Searched {topics_count} topics in {elapsed} seconds.")
        if topics_count != expected_topics:
            print(f"!!! Warning: Expected {expected_topics} topics, but processed {topics_count}.")

    def close(self):
        self.reader.close()

# main
if __name__ == "__main__":
    
    # 1. Define all absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "index"))
    RUNS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "runs"))
    DATA_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))
    JOURNALS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Parse", "journals.txt"))

    similarity = BM25Similarity(0.9, 0.8)
    # Initialize analyzers
    analyzer_map = HashMap()
    
    # English
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_en", EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_en", EnglishAnalyzer())
    
    # French
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_fr", FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_fr", FrenchAnalyzer())
    
    # German
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_de", GermanAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_de", GermanAnalyzer())

    multi_analyzer = PerFieldAnalyzerWrapper(StandardAnalyzer(), analyzer_map)
    # Initialize the Searcher
    engine = Searcher(
        analyzer=multi_analyzer,
        similarity=similarity,
        index_path=INDEX_PATH,
        run_id="run-native-analyzers-prf", # Updated run name
        run_path=RUNS_PATH,
        max_docs_retrieved=1000  
    )

    claim_parser = ClaimParser(data_dir=DATA_FOLDER, journals_file=JOURNALS_PATH)
    
    tasks = [
        {"file": "en_dev.jsonl", "lang": "en", "expected": 195},
        {"file": "de_dev.jsonl", "lang": "de", "expected": 100},
        {"file": "fr_dev.jsonl", "lang": "fr", "expected": 100}
    ]

    try:
        for task in tasks:
            print(f"Elaborazione del file: {task['file']}")
            queries_iterator = claim_parser.parse_language_file(task["file"], task["lang"])
            engine.search_topics(queries_iterator, task["expected"])
                
    finally:
        engine.close()
        print(f"\n#### All Searching Complete! Results saved to {RUNS_PATH} ####")