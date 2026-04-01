import lucene # need to import otherwise ModuleNotFoundError: No module named 'org'
#from java.io import File 

import sys
import time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) #adding project root path
# Core Lucene indexing tools
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths

# Lucene Documents and Fields
from org.apache.lucene.document import Document, Field, StringField, FieldType

# Text Analysis and Scoring
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
#from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

## verify that pyLucene is working and initialize the environment
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

# until here is how the professor did
from src.Parse.parse_documents import ParsedDocument, JsonArrayDocumentParser
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



class BodyFieldType:
    """
    Replicates the static configuration of BodyField.java.
    We create a singleton FieldType to reuse across all documents.
    """

    TYPE = FieldType()
    TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS) 
    TYPE.setTokenized(True) # Chops the abstract/title into individual words
    TYPE.setStored(True) # Do not store raw text to save space (SAVING NOW for Luke debugging)
    TYPE.freeze() # Prevents future changes

    @staticmethod
    def create(value: str):
        return Field(ParsedDocument.FIELDS.BODY, value, BodyFieldType.TYPE)

class DirectoryIndexer:
    """
    Indexes documents processing a whole directory tree.
    Replicates the Professor's DirectoryIndexer.java.
    """
    MBYTE = 1024 * 1024

    def __init__(self, analyzer, similarity, ram_buffer_size_mb: int,
                 index_path: str, docs_path: str, extension: str, 
                 expected_docs: int, parser_class):
        
    #----------- Validation checks -------------------------------
        if ram_buffer_size_mb <= 0:
            raise ValueError("RAM buffer size must be > 0.")
        if not index_path or not docs_path:
            raise ValueError("Paths cannot be empty.")
        
        self.index_dir = index_path
        self.docs_path = docs_path
        self.extension = extension
        self.expected_docs = expected_docs
        self.parser_class = parser_class
        
    #-----------  Ensure input directory exists ----------- 
        if not os.path.isdir(docs_path):
            raise ValueError(f"Documents directory {docs_path} does not exist.")
            
        # Ensure output directory exists
        os.makedirs(index_path, exist_ok=True)
        #self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize IndexWriter
        iwc = IndexWriterConfig(analyzer)
        # Configure IndexWriter
        iwc.setSimilarity(similarity)
        iwc.setRAMBufferSizeMB(float(ram_buffer_size_mb))
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        iwc.setCommitOnClose(True)
        iwc.setUseCompoundFile(False)

        # Initialize tracking variables, this is to track metrics after the index process
        self.docs_count = 0
        self.bytes_count = 0
        self.files_count = 0
        self.start_time = time.time()

        # Open the Lucene Directory and create Writer
        java_index_dir = Paths.get(index_path)
        fs_dir = FSDirectory.open(java_index_dir)
        self.writer = IndexWriter(fs_dir, iwc)

    def index(self):
        print("\n#### Start indexing ####")
        
        # Walk through the directory (Python equivalent of Files.walkFileTree)
        for root, dirs, files in os.walk(self.docs_path):
            for file_name in files:
                if file_name.endswith(self.extension):
                    file_path = os.path.join(root, file_name)
                    self.files_count += 1
                    self.bytes_count += os.path.getsize(file_path)
                    
                    # Open file and instantiate the parser
                    with open(file_path, 'r', encoding='utf-8') as f:
                        parser = self.parser_class(f)
                        
                        # Iterate through parsed documents
                        for pd in parser:
                            doc = Document()
                            
                            # Field.Store.YES means we can retrieve this ID later
                            # Add ID Field and Lang Field
                            doc.add(StringField(ParsedDocument.FIELDS.ID, pd.id, Field.Store.YES))
                            doc.add(StringField("lang", getattr(pd, "lang", "en"), Field.Store.YES))
                            
                            # Add TITLE and ABSTRACT (Usiamo lo stesso BodyFieldType perché vogliamo le stesse regole per entrambi)
                            # doc.add(Field(ParsedDocument.FIELDS.TITLE, pd.title, BodyFieldType.TYPE))
                            # doc.add(Field(ParsedDocument.FIELDS.ABSTRACT, pd.abstract, BodyFieldType.TYPE))
                            
                            # Creiamo i campi dinamici in base alla lingua (es. TITLE_en, ABSTRACT_fr)
                            lang = getattr(pd, "lang", "en") # Assicuriamoci che ci sia sempre una lingua
                            title_field = f"{ParsedDocument.FIELDS.TITLE}_{lang}"
                            abstract_field = f"{ParsedDocument.FIELDS.ABSTRACT}_{lang}"
                            
                            # Replace "-" with ""
                            title_no_hyphens = pd.title.replace("-", "") if pd.title else ""
                            abstract_no_hyphens = pd.abstract.replace("-", "") if pd.abstract else ""

                            doc.add(Field(title_field, title_no_hyphens, BodyFieldType.TYPE))
                            doc.add(Field(abstract_field, abstract_no_hyphens, BodyFieldType.TYPE))
                            
                            # Write to index
                            self.writer.addDocument(doc)
                            self.docs_count += 1
                            
                            # Print progress every 10,000 docs
                            if self.docs_count % 1000 == 0:
                                elapsed = int(time.time() - self.start_time)
                                mb_indexed = self.bytes_count // self.MBYTE
                                print(f"{self.docs_count} document(s) ({self.files_count} files, "
                                      f"{mb_indexed} Mbytes) indexed in {elapsed} seconds.")

        # Commit and close the index
        self.writer.commit()
        self.writer.close()

        # Final Summary
        if self.docs_count != self.expected_docs:
            print(f"Warning: Expected to index {self.expected_docs} docs; {self.docs_count} indexed instead.")
            
        total_elapsed = int(time.time() - self.start_time)
        total_mb = self.bytes_count // self.MBYTE
        print(f"{self.docs_count} document(s) ({self.files_count} files, "
              f"{total_mb} Mbytes) indexed in {total_elapsed} seconds.")
        print("#### Indexing complete ####\n")


from src.Parse.parse_documents import JsonArrayDocumentParser
if __name__ == "__main__":
    
    
    DOCS_PATH = "./experiment/docs/" # Folder where your collection.jsonl is
    INDEX_PATH = "./experiment/index" # Where Lucene will save the database
    
    
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

    # 3. Initialize the Indexer
    # We pass the CollectionParser class we built previously
    try:
        indexer = DirectoryIndexer(
            analyzer=multi_analyzer,
            similarity=similarity,
            ram_buffer_size_mb=256,
            index_path=INDEX_PATH,
            docs_path=DOCS_PATH,
            extension=".jsonl",
            expected_docs=34237,
            parser_class=JsonArrayDocumentParser # <--- Replace with your actual parser class name
        )
        
        # 4. Run the engine
        indexer.index()
        
    except Exception as e:
        print(f"An error occurred during indexing: {e}")