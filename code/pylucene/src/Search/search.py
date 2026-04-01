import lucene
import os
import sys
import time
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
#from src.Parse.parse_query import ScientificClaimQuery, load_journals
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
#from src.Parse.TextProcessor import TextProcessor
#processor = TextProcessor()


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
        # 2. Setup the Query Parser (Search in both TITLE and ABSTRACT)
        fields_to_search = [ParsedDocument.FIELDS.TITLE, ParsedDocument.FIELDS.ABSTRACT]
        
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

        # Open the run file in 'a' (append) mode so we can add EN, DE, and FR sequentially
        with open(self.run_file_path, 'a', encoding='utf-8') as run_file:
            
            for topic in parser:
                topics_count += 1
                bq_builder = BooleanQuery.Builder()

                # --- NOVITÀ: Costruiamo il Parser in base alla lingua della query ---
                lang = topic.language
                fields_to_search = [f"{ParsedDocument.FIELDS.TITLE}_{lang}", f"{ParsedDocument.FIELDS.ABSTRACT}_{lang}"]

                # Passiamo i campi dinamici al QueryParser
                qp = MultiFieldQueryParser(fields_to_search, self.analyzer)
                # --- A. THE MAIN QUERY (The cleaned tweet) ---
                clean_text = topic.get_search_query()
                if clean_text:
                    # Stemming the queries
                    # stemmed_tokens = processor.process(clean_text, topic.language)
                    # stemmed_text = " ".join(stemmed_tokens)
                    # Escape special characters so Lucene doesn't crash on weird math symbols
                    escaped_title = QueryParserBase.escape(clean_text)
                    title_query = QueryParserBase.parse(qp, escaped_title)
                    # SHOULD means "Match these words if possible"
                    bq_builder.add(title_query, BooleanClause.Occur.SHOULD)

                # --- B. THE BOOST QUERY (The extracted journals) ---
                journals_found = topic.extract_entities()
                if journals_found:
                    journals_text = " ".join(journals_found) # Tranform the list in a string
                    escaped_desc = QueryParserBase.escape(journals_text)
                    desc_query = QueryParserBase.parse(self.qp, escaped_desc)
                    # Adding this as another SHOULD clause naturally boosts the BM25 score 
                    # if the document contains the journal name!
                    bq_builder.add(desc_query, BooleanClause.Occur.SHOULD)
                
                # --- C. THE LANGUAGE FILTER (The CheckThat! requirement) ---
                # MUST means "If the document is not this language, drop it instantly"
                lang_term = TermQuery(Term("lang", topic.language))
                bq_builder.add(lang_term, BooleanClause.Occur.MUST)

                # Finalize the query
                final_query = bq_builder.build()

                # --- D. EXECUTE SEARCH ---
                top_docs = self.searcher.search(final_query, self.max_docs)
                score_docs = top_docs.scoreDocs

                # --- E. WRITE TO TREC FORMAT ---
                for rank, sd in enumerate(score_docs):
                    doc = stored_fields.document(sd.doc)
                    doc_id = doc.get(ParsedDocument.FIELDS.ID)
                    
                    # Format: query_id Q0 doc_id rank score run_id
                    line = f"{topic.query_id}\tQ0\t{doc_id}\t{rank}\t{sd.score:.6f}\t{self.run_id}\n"
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

    #analyzer = WhitespaceAnalyzer()
    similarity = BM25Similarity()
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
        run_id="run-native-analyzers",
        run_path=RUNS_PATH,
        max_docs_retrieved=1000  # TREC standard requires top 1000
    )

    # Load Journals
    #journals_list = ClaimParser.load_journals(filepath =JOURNALS_PATH)
    # since there is an error I call it as it is an utility function
    
    #journals_list = load_journals(JOURNALS_PATH)
    claim_parser = ClaimParser(data_dir=DATA_FOLDER, journals_file=JOURNALS_PATH)
    
    # Define the batches of queries to process
    tasks = [
        {"file": "en_dev.jsonl", "lang": "en", "expected": 195},
        {"file": "de_dev.jsonl", "lang": "de", "expected": 100},
        {"file": "fr_dev.jsonl", "lang": "fr", "expected": 100}
    ]

    '''try:
        # loop through languages and search!
        for task in tasks:
            file_path = os.path.join(DATA_FOLDER, task["file"])
            if not os.path.exists(file_path):
                print(f"Skipping {task['file']} - file not found.")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                parser = ScientificClaimQuery(f, journals_list=journals_list, lang_code=task["lang"])
                engine.search_topics(parser, task["expected"])
                
    finally:
        engine.close()
        print(f"\n#### All Searching Complete! Results saved to {RUNS_PATH} ####")'''
    try:
        # loop through languages and search!
        for task in tasks:
            print(f"Elaborazione del file: {task['file']}")
            
            # Usiamo ClaimParser per estrarre la lista delle query dal file
            queries_iterator = claim_parser.parse_language_file(task["file"], task["lang"])
            
            # Passiamo l'iteratore pieno di query al motore di ricerca
            engine.search_topics(queries_iterator, task["expected"])
                
    finally:
        engine.close()
        print(f"\n#### All Searching Complete! Results saved to {RUNS_PATH} ####")