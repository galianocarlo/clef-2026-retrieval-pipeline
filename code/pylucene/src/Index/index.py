import lucene  # need to import otherwise ModuleNotFoundError: No module named 'org'
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Core Lucene indexing tools
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
from java.util import HashMap

# Lucene Documents and Fields
from org.apache.lucene.document import Document, Field, StringField, FieldType

# Text Analysis and Scoring
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.fr import FrenchAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer

# Verify that PyLucene is working and initialize the JVM
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

from src.Parse.parse_documents import ParsedDocument, JsonArrayDocumentParser


class BodyFieldType:
    """
    Singleton FieldType shared across every title/abstract field.

    DOCS_AND_FREQS_AND_POSITIONS is used instead of the old DOCS_AND_FREQS so
    that phrase queries (e.g. "machine learning") and proximity queries work at
    search time.  The index grows ~15-20 % larger, but exact-phrase precision
    is worth the cost for a scientific-paper corpus.
    """

    TYPE = FieldType()
    TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    TYPE.setTokenized(True)   # tokenise title / abstract into individual terms
    TYPE.setStored(True)      # store raw text so Luke and stored-field retrieval work
    TYPE.freeze()             # prevent accidental mutation after class load


class DirectoryIndexer:
    """
    Indexes a document collection stored as one or more JSONL files inside a
    directory tree.  Replicates the professor's DirectoryIndexer.java in Python.
    """

    MBYTE = 1024 * 1024

    def __init__(
        self,
        analyzer,
        similarity,
        ram_buffer_size_mb: int,
        index_path: str,
        docs_path: str,
        extension: str,
        expected_docs: int,
        parser_class,
        include_files: set = None,
    ):
        """
        Parameter:
            analyzer: Lucene Analyzer
            similarity: Lucene Similarity
            ram_buffer_size_mb: IndexWriter RAM buffer in MB
            index_path: directory where Lucene writes the index
            docs_path: root directory that is walked for JSONL files
            extension: file extension to look for (e.g. ".jsonl")
            expected_docs: expected document count — triggers a warning if the actual count differs
            parser_class: iterable parser class; called as parser_class(file)
            include_files: optional set of *basenames* to index (e.g. {"collection.jsonl"}). If provided, every other matching file is skipped.
                                    This prevents topic / query files that share the same directory and extension from being indexed.
                                    Pass None to index every file that matches extension.
        """
        # --- Validation ---------------------------------------------------
        if ram_buffer_size_mb <= 0:
            raise ValueError("RAM buffer size must be > 0.")
        if not index_path or not docs_path:
            raise ValueError("Paths cannot be empty.")
        if not os.path.isdir(docs_path):
            raise ValueError(f"Documents directory '{docs_path}' does not exist.")

        # --- Store config -------------------------------------------------
        self.docs_path     = docs_path
        self.extension     = extension
        self.expected_docs = expected_docs
        self.parser_class  = parser_class
        self.include_files = include_files   # None → no filter

        # --- Ensure output directory exists -------------------------------
        os.makedirs(index_path, exist_ok=True)

        # --- Set up IndexWriter -------------------------------------------
        iwc = IndexWriterConfig(analyzer)
        iwc.setSimilarity(similarity)
        iwc.setRAMBufferSizeMB(float(ram_buffer_size_mb))
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        iwc.setCommitOnClose(True)
        iwc.setUseCompoundFile(False)

        # --- Counters for progress reporting ------------------------------
        self.docs_count  = 0
        self.bytes_count = 0
        self.files_count = 0
        self.start_time  = time.time()

        # --- Open Lucene FSDirectory and create the writer ----------------
        fs_dir = FSDirectory.open(Paths.get(index_path))
        self.writer = IndexWriter(fs_dir, iwc)

    def index(self):
        """Walk docs_path and index every qualifying file."""
        print("\n#### Start indexing ####")

        for root, _dirs, files in os.walk(self.docs_path):
            for file_name in files:

                # 1. Extension filter
                if not file_name.endswith(self.extension):
                    continue

                # 2. Whitelist filter — skips query / topic files that happen
                #    to live in the same folder as the paper collection.
                if self.include_files is not None and file_name not in self.include_files:
                    print(f"  [skip] {file_name}  (not in include_files)")
                    continue

                file_path = os.path.join(root, file_name)
                self.files_count += 1
                self.bytes_count += os.path.getsize(file_path)

                with open(file_path, "r", encoding="utf-8") as f:
                    parser = self.parser_class(f)

                    for pd in parser:
                        doc = Document()

                        # --- Stored fields (needed to recover the id later) ---
                        doc.add(StringField(ParsedDocument.FIELDS.ID, pd.id, Field.Store.YES))
                        doc.add(StringField("lang", getattr(pd, "lang", "en"), Field.Store.YES))

                        # --- Language-specific analysed fields ---------------
                        # e.g. title_en / abstract_fr / title_de …
                        lang           = getattr(pd, "lang", "en")
                        title_field    = f"{ParsedDocument.FIELDS.TITLE}_{lang}"
                        abstract_field = f"{ParsedDocument.FIELDS.ABSTRACT}_{lang}"

                        # Guard: Field() raises a Java exception when value is None
                        doc.add(Field(title_field,    pd.title    or "", BodyFieldType.TYPE))
                        doc.add(Field(abstract_field, pd.abstract or "", BodyFieldType.TYPE))

                        self.writer.addDocument(doc)
                        self.docs_count += 1

                        # Progress every 1 000 documents
                        if self.docs_count % 1000 == 0:
                            elapsed    = int(time.time() - self.start_time)
                            mb_indexed = self.bytes_count // self.MBYTE
                            print(
                                f"{self.docs_count} document(s) "
                                f"({self.files_count} files, {mb_indexed} MB) "
                                f"indexed in {elapsed}s."
                            )

        # --- Finalise -----------------------------------------------------
        self.writer.commit()
        self.writer.close()

        if self.docs_count != self.expected_docs:
            print(
                f"Warning: expected {self.expected_docs} docs, "
                f"indexed {self.docs_count}."
            )

        total_elapsed = int(time.time() - self.start_time)
        total_mb      = self.bytes_count // self.MBYTE
        print(
            f"{self.docs_count} document(s) ({self.files_count} files, "
            f"{total_mb} MB) indexed in {total_elapsed}s."
        )
        print("#### Indexing complete ####\n")


if __name__ == "__main__":

    #  Paths
    DOCS_PATH  = "./experiment/docs/"  # contains collection.jsonl + topic files
    INDEX_PATH = "./experiment/index"  # Lucene writes its index here

    # Only these basenames will be indexed; every other .jsonl is skipped.
    COLLECTION_FILES = {"collection.jsonl"}

    #  BM25 hyper-parameters
    BM25_K1    = 0.9
    BM25_B     = 0.8
    similarity = BM25Similarity(BM25_K1, BM25_B)

    #  Per-field analyzers                                                 #
    analyzer_map = HashMap()

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_en",       EnglishAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_en",    EnglishAnalyzer())

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_fr",       FrenchAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_fr",    FrenchAnalyzer())

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_de",       GermanAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_de",    GermanAnalyzer())

    analyzer_map.put(f"{ParsedDocument.FIELDS.TITLE}_other",    StandardAnalyzer())
    analyzer_map.put(f"{ParsedDocument.FIELDS.ABSTRACT}_other", StandardAnalyzer())

    multi_analyzer = PerFieldAnalyzerWrapper(StandardAnalyzer(), analyzer_map)

    #  Run
    try:
        indexer = DirectoryIndexer(
            analyzer=multi_analyzer,
            similarity=similarity,
            ram_buffer_size_mb=256,
            index_path=INDEX_PATH,
            docs_path=DOCS_PATH,
            extension=".jsonl",
            expected_docs=10000,
            parser_class=JsonArrayDocumentParser,
            include_files=COLLECTION_FILES
        )
        indexer.index()

    except Exception as e:
        print(f"An error occurred during indexing: {e}")
