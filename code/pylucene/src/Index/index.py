import lucene # need to import otherwise ModuleNotFoundError: No module named 'org'
#from java.io import File 

import sys
import time
import os
# Core Lucene indexing tools
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths

# Lucene Documents and Fields
from org.apache.lucene.document import Document, Field, StringField, FieldType

# Text Analysis and Scoring
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

## verify that pyLucene is working and initialize the environment
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

# until here is how the professor did

"""# ---------------------------------------------------------
# Assuming ParsedDocument and CollectionParser are defined here
# (Paste the classes we built earlier here if they aren't in another file)
class ParsedDocument:
    '''A simplified 'Standard Box' just for testing the indexer.'''
    def __init__(self, doc_id: str, body: str):
        self.identifier = str(doc_id)
        self.body = str(body)

    def getBody(self):
        return self.body

class DummyParser:
    
    '''A fake worker that pretends to read a file, but actually 
    just spits out 3 fake documents so you can test Lucene.'''
    
    def __init__(self, file_reader):
        # We accept the file_reader so the DirectoryIndexer doesn't crash, 
        # but we completely ignore it for this test.
        self.fake_data = [
            ParsedDocument("doc_1", "The quick brown fox jumps over the lazy dog."),
            ParsedDocument("doc_2", "SARS-CoV-2 and COVID-19 research is ongoing."),
            ParsedDocument("doc_3", "Apache Lucene makes building search engines easy.")
        ]
        # Create an iterator out of our fake list
        self._iterator = iter(self.fake_data)

    def __iter__(self):
        return self

    def __next__(self):
        # Spits out one fake document at a time until the list is empty
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopIteration()
# ---------------------------------------------------------
"""
class BodyFieldType:
    """
    Replicates the static configuration of BodyField.java.
    We create a singleton FieldType to reuse across all documents.
    """

    TYPE = FieldType()
    TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS) 
    TYPE.setTokenized(True) # Chops the abstract/title into individual words
    TYPE.setStored(False) # Do not store raw text to save space
    TYPE.freeze() # Prevents future changes

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
                            
                            # Add ID Field (Stored)
                            # Field.Store.YES means we can retrieve this ID later
                            doc.add(StringField("id", pd.id, Field.Store.YES))
                            
                            # Add Body Field (Not Stored, only indexed)
                            doc.add(Field("body", pd.body, BodyFieldType.TYPE))
                            
                            # Write to index
                            self.writer.addDocument(doc)
                            self.docs_count += 1
                            
                            # Print progress every 10,000 docs
                            if self.docs_count % 10000 == 0:
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


from parse_documents import JsonArrayDocumentParser
if __name__ == "__main__":
    # 1. Setup paths and parameters
    
    DOCS_PATH = "./experiment/docs" # Folder where your collection.jsonl is
    INDEX_PATH = "./experiment/index" # Where Lucene will save the database
    
    # 2. Configure Analyzer and Similarity
    # StandardAnalyzer handles tokenization, lowercasing, and basic stopword removal
    analyzer = StandardAnalyzer()
    similarity = BM25Similarity()
    
    # 3. Initialize the Indexer
    # We pass the CollectionParser class we built previously
    try:
        indexer = DirectoryIndexer(
            analyzer=analyzer,
            similarity=similarity,
            ram_buffer_size_mb=256,
            index_path=INDEX_PATH,
            #docs_path=DOCS_PATH, # Make sure this folder exists, even if it has empty text files!
            #extension=".txt",    # Look for any .txt file just to trigger the loop
            #expected_docs=3,     # We expect 3 fake documents
            #parser_class=DummyParser # <--- USE THE FAKE PARSER HERE
            docs_path=DOCS_PATH,
            extension=".jsonl",
            expected_docs=100000, # Put your expected collection size here
            parser_class=JsonArrayDocumentParser # <--- Replace with your actual parser class name
        )
        
        # 4. Run the engine
        indexer.index()
        
    except Exception as e:
        print(f"An error occurred during indexing: {e}")