import lucene
import os
import sys
import time
from java.nio.file import Paths
from java.util import HashMap
from java.lang import Float

from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.queryparser.classic import QueryParser, QueryParserBase, MultiFieldQueryParser

from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.Parse.parse_documents import ParsedDocument
from src.Parse.parse_query import ClaimParser
from src.Parse.TextProcessor import TextProcessor

processor = TextProcessor()

assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

class Searcher:
    def __init__(self, analyzer, similarity, index_path: str, run_id: str, run_path: str, max_docs_retrieved: int):
        
        self.index_dir = Paths.get(index_path)
        self.fs_dir = FSDirectory.open(self.index_dir)
        self.reader = DirectoryReader.open(self.fs_dir)
        
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(similarity)

        fields_to_search = [ParsedDocument.FIELDS.TITLE, ParsedDocument.FIELDS.ABSTRACT]
        
        boosts = HashMap()
        boosts.put(ParsedDocument.FIELDS.TITLE, Float(1.0))
        boosts.put(ParsedDocument.FIELDS.ABSTRACT, Float(1.0))
        
        self.qp = MultiFieldQueryParser(fields_to_search, analyzer, boosts)
        
        self.run_id = run_id
        self.max_docs = max_docs_retrieved
        
        os.makedirs(run_path, exist_ok=True)
        self.run_file_path = os.path.join(run_path, f"{run_id}.txt")
        
        with open(self.run_file_path, 'w', encoding='utf-8') as f:
            pass 

    def search_topics(self, parser, expected_topics: int):
        print(f"\n#### Start searching {expected_topics} topics ####")
        start_time = time.time()
        topics_count = 0
        
        stored_fields = self.searcher.storedFields()

        with open(self.run_file_path, 'a', encoding='utf-8') as run_file:
            
            for topic in parser:
                topics_count += 1
                bq_builder = BooleanQuery.Builder()

                clean_text = topic.get_search_query()
                if clean_text:
                    expanded_tokens = processor.expand_query(clean_text, topic.language)
                    expanded_text = " ".join(expanded_tokens)
                    escaped_query = QueryParserBase.escape(expanded_text)
                    main_query = QueryParserBase.parse(self.qp, escaped_query)
                    bq_builder.add(main_query, BooleanClause.Occur.SHOULD)

                journals_found = topic.extract_entities()
                if journals_found:
                    journals_text = " ".join(journals_found) 
                    escaped_desc = QueryParserBase.escape(journals_text)
                    desc_query = QueryParserBase.parse(self.qp, escaped_desc)
                    bq_builder.add(desc_query, BooleanClause.Occur.SHOULD)
                
                lang_term = TermQuery(Term("lang", topic.language))
                bq_builder.add(lang_term, BooleanClause.Occur.MUST)

                final_query = bq_builder.build()

                top_docs = self.searcher.search(final_query, self.max_docs)
                score_docs = top_docs.scoreDocs

                for rank, sd in enumerate(score_docs):
                    doc = stored_fields.document(sd.doc)
                    doc_id = doc.get(ParsedDocument.FIELDS.ID)
                    
                    line = f"{topic.query_id}\tQ0\t{doc_id}\t{rank}\t{sd.score:.6f}\t{self.run_id}\n"
                    run_file.write(line)

        elapsed = int(time.time() - start_time)
        print(f"Searched {topics_count} topics in {elapsed} seconds.")
        if topics_count != expected_topics:
            print(f"!!! Warning: Expected {expected_topics} topics, but processed {topics_count}.")

    def close(self):
        self.reader.close()

if __name__ == "__main__":
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "index"))
    RUNS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "runs"))
    DATA_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))
    JOURNALS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Parse", "journals.txt"))

    analyzer = WhitespaceAnalyzer()
    similarity = BM25Similarity()
    
    engine = Searcher(
        analyzer=analyzer,
        similarity=similarity,
        index_path=INDEX_PATH,
        run_id="seupd-baseline-run",
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