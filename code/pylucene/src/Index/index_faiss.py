import lucene
import sys
import time
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
from org.apache.lucene.document import Document, Field, StringField, FieldType
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

assert lucene.getVMEnv() or lucene.initVM()

from src.Parse.parse_documents import ParsedDocument, JsonArrayDocumentParser
from java.util import HashMap
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.fr import FrenchAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer

class BodyFieldType:
    TYPE = FieldType()
    TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS) 
    TYPE.setTokenized(True)
    TYPE.setStored(True)
    TYPE.freeze()

    @staticmethod
    def create(value: str):
        return Field(ParsedDocument.FIELDS.BODY, value, BodyFieldType.TYPE)

class DirectoryIndexer:
    MBYTE = 1024 * 1024

    def __init__(self, analyzer, similarity, ram_buffer_size_mb: int,
                 index_path: str, docs_path: str, extension: str, 
                 expected_docs: int, parser_class):
        
        if ram_buffer_size_mb <= 0:
            raise ValueError()
        if not index_path or not docs_path:
            raise ValueError()
        
        self.index_dir = index_path
        self.docs_path = docs_path
        self.extension = extension
        self.expected_docs = expected_docs
        self.parser_class = parser_class
        
        if not os.path.isdir(docs_path):
            raise ValueError()
            
        os.makedirs(index_path, exist_ok=True)

        iwc = IndexWriterConfig(analyzer)
        iwc.setSimilarity(similarity)
        iwc.setRAMBufferSizeMB(float(ram_buffer_size_mb))
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        iwc.setCommitOnClose(True)
        iwc.setUseCompoundFile(False)

        self.docs_count = 0
        self.bytes_count = 0
        self.files_count = 0
        self.start_time = time.time()

        java_index_dir = Paths.get(index_path)
        fs_dir = FSDirectory.open(java_index_dir)
        self.writer = IndexWriter(fs_dir, iwc)

        self.faiss_index_path = os.path.join(self.index_dir, "corpus.faiss")
        self.doc_ids_path = os.path.join(self.index_dir, "doc_ids.json")
        self.batch_size = 32
        
        self.encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dim = self.encoder.get_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_doc_ids = []
        
        self.text_batch = []
        self.id_batch = []

    def _process_faiss_batch(self):
        if not self.text_batch:
            return
        
        embeddings = self.encoder.encode(self.text_batch, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        self.faiss_index.add(embeddings)
        self.faiss_doc_ids.extend(self.id_batch)
        
        self.text_batch = []
        self.id_batch = []

    def index(self):
        for root, dirs, files in os.walk(self.docs_path):
            for file_name in files:
                if file_name.endswith(self.extension):
                    file_path = os.path.join(root, file_name)
                    self.files_count += 1
                    self.bytes_count += os.path.getsize(file_path)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        parser = self.parser_class(f)
                        
                        for pd in parser:
                            doc = Document()
                            
                            doc.add(StringField(ParsedDocument.FIELDS.ID, pd.id, Field.Store.YES))
                            doc.add(StringField("lang", getattr(pd, "lang", "en"), Field.Store.YES))
                            
                            lang = getattr(pd, "lang", "en")
                            title_field = f"{ParsedDocument.FIELDS.TITLE}_{lang}"
                            abstract_field = f"{ParsedDocument.FIELDS.ABSTRACT}_{lang}"
                            
                            title_clean = pd.title if pd.title else ""
                            abstract_clean = pd.abstract if pd.abstract else ""

                            # 1. Lucene Indexing (BM25)
                            doc.add(Field(title_field, title_clean, BodyFieldType.TYPE))
                            doc.add(Field(abstract_field, abstract_clean, BodyFieldType.TYPE))
                            self.writer.addDocument(doc)

                            # 2. Dense Indexing (FAISS)
                            full_text = f"{title_clean}. {abstract_clean}"
                            self.text_batch.append(full_text)
                            self.id_batch.append(pd.id)

                            if len(self.text_batch) >= self.batch_size:
                                self._process_faiss_batch()

                            self.docs_count += 1
                            
                            if self.docs_count % 1000 == 0:
                                elapsed = int(time.time() - self.start_time)
                                mb_indexed = self.bytes_count // self.MBYTE
                                print(f"{self.docs_count} document(s) ({self.files_count} files, "
                                      f"{mb_indexed} Mbytes) indexed in {elapsed} seconds.")

        self._process_faiss_batch()

        self.writer.commit()
        self.writer.close()

        faiss.write_index(self.faiss_index, self.faiss_index_path)
        with open(self.doc_ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.faiss_doc_ids, f)

        if self.docs_count != self.expected_docs:
            print(f"Warning: Expected {self.expected_docs} docs; {self.docs_count} indexed.")
            
        total_elapsed = int(time.time() - self.start_time)
        total_mb = self.bytes_count // self.MBYTE
        print(f"{self.docs_count} document(s) ({self.files_count} files, "
              f"{total_mb} Mbytes) indexed in {total_elapsed} seconds.")

if __name__ == "__main__":
    DOCS_PATH = "./experiment/docs/"
    INDEX_PATH = "./experiment/index"
    
    similarity = BM25Similarity(0.9, 0.8)
    analyzer_map = HashMap()
    
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_en", EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_en", EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_fr", FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_fr", FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_de", GermanAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_de", GermanAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_other", StandardAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_other", StandardAnalyzer())
    
    multi_analyzer = PerFieldAnalyzerWrapper(StandardAnalyzer(), analyzer_map)

    try:
        indexer = DirectoryIndexer(
            analyzer=multi_analyzer,
            similarity=similarity,
            ram_buffer_size_mb=256,
            index_path=INDEX_PATH,
            docs_path=DOCS_PATH,
            extension=".jsonl",
            expected_docs=34237,
            parser_class=JsonArrayDocumentParser
        )
        indexer.index()
    except Exception as e:
        print(e)