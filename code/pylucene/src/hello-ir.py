# Copyright 2026-2026 University of Padua, Italy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Hello IR in pyLucene

Introductory example on how to use <a href="https://lucene.apache.org/pylucene/" target="_blank">Apache PyLucene</a>
to index and search for toy documents.

@author Nicola Ferro (nicola.ferro@unipd.it)
@version 1.0
@since 1.0
"""

import sys

import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader
from org.apache.lucene.queryparser.simple import SimpleQueryParser
from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.store import FSDirectory

## verify that pyLucene is working and initialize the environment
assert lucene.getVMEnv() or lucene.initVM(), "Lucene version not available"

# The first toy document.
# Credits to Maria Maistro.
D1 = "The quokka, the only member of the genus Setonix, is a small marsupial about the size of a domestic cat"

# The second toy document.
# Credits to Maria Maistro.
D2 = "Wombats are small, short-legged, muscular quadrupedal marsupials that are native to Australia"

# The third toy document.
# Credits to Maria Maistro.
D3 = "Quokkas have little fear of humans and commonly approach people closely, particularly on Rottnest Island in Australia, where a prevalent population exists"

# The name of the id Field in a Document.
ID = "id"

# The name of the body Field in a Document.
BODY = "body"

# The default path to the directory containing the index
DEFAULT_INDEX_PATH = "experiment/index/pylucene"

# The default run file
DEFAULT_RUN_FILE = "experiment/runs/runs-pylucene.txt" # modofied dor the project

# The identifier of the run to be used in the run file.
RUN_ID = "hello-IR-pylucene"

# The maximum number of documents to be retrieved for each query
MAX_DOCS_RETRIEVED = 10


def parse_documents():
    """Parses the toy documents and returns a list of Documents.

        Returns:
            list: a list of org.apache.lucene.Document objects
    """

    print("------------- PARSING DOCUMENTS -------------")

    # the list of returned documents
    docs = []

    # create a new document for D1
    d = Document()
    d.add(Field(ID, "d1", TextField.TYPE_STORED))
    d.add(Field(BODY, D1, TextField.TYPE_STORED))
    docs.append(d)

    # create a new document for D2
    d = Document()
    d.add(Field(ID, "d2", TextField.TYPE_STORED))
    d.add(Field(BODY, D2, TextField.TYPE_STORED))
    docs.append(d)

    # create a new document for D3
    d = Document()
    d.add(Field(ID, "d3", TextField.TYPE_STORED))
    d.add(Field(BODY, D3, TextField.TYPE_STORED))
    docs.append(d)

    print("The documents are:")
    for d in docs:
        print(d)

    print("---------------------------------------------\n")

    return docs


def index(index_path, docs, analyzer):
    """Indexes the provided Documents with the given Analyzer.

        Args:
            index_path (str): The path where to store the index.
            docs (list): a list of documents to be indexed.
            analyzer (Analyzer): the analyzer to use for indexing.
    """


    print("------------- INDEXING DOCUMENTS -------------")

    # Open the directory in Lucene
    directory = FSDirectory.open(File(index_path).toPath())

    # Hold all the required configuration for the indexer
    config = IndexWriterConfig(analyzer)

    # force to re-create the index if it already exists
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    # set the similarity. BM25 is already the default one
    config.setSimilarity(BM25Similarity())

    # The actual indexer
    writer = IndexWriter(directory, config)

    print("- Indexer successfully created")

    for d in docs:
        writer.addDocument(d)

        print(f"- Document {d.get(ID)} successfully indexed")

    writer.close()
    directory.close()
    print("- Indexer successfully closed")

    print("----------------------------------------------\n")


def search(index_path, query_id, query, analyzer, results):
    """Searches for the provided query, by processing it with the given Analyzer.

        Args:
            index_path (str): The path where to store the index.
            query_id (str): The id of the query to search for.
            query (str): The query to search for.
            analyzer (Analyzer): the analyzer to use for searching.
            results (list): a list of results for the query in trec_eval format.


        Returns:
            list: the update list of results in trec_eval format.
    """

    print(f"------------- SEARCHING DOCUMENTS FOR QUERY {query_id} -------------")

    # Open the directory in Lucene
    directory = FSDirectory.open(File(index_path).toPath())

    # Reads the index
    reader = DirectoryReader.open(directory)

    # Searches the index
    searcher = IndexSearcher(reader)

    # Set the similarity. BM25 is already the default one
    searcher.setSimilarity(BM25Similarity())

    print("- Searcher successfully created")

    # Parses the textual query
    qp = SimpleQueryParser(analyzer, BODY)

    q = qp.parse(query)
    print(f"- Query successfully parsed: {q}")

    # Perform the actual search
    hits = searcher.search(q, MAX_DOCS_RETRIEVED).scoreDocs
    print(f"- {len(hits)} document(s) retrieved")

    # Used to read documents from the index
    stored_fields = searcher.storedFields()

    for i in range(len(hits)):
        ## Read the document from the index
        doc = stored_fields.document(hits[i].doc)

        ## append the result to the list of results
        results.append(f"{query_id}\tQ0\t{doc.get(ID)}\t{i}\t{hits[i].score:.6f}\t{RUN_ID}")

        print(results[i])

    reader.close()
    directory.close()

    print("- Searcher successfully closed")
    print("-------------------------------------------------------------\n")

    return results


def write_run_file(results, run_file):
    """Writes the results to a run file.

        Args:
            results (list): a list of results for one or more queries in trec_eval format.
            run_file (str): the path to the run file.
    """

    run = open(run_file, "w")

    for r in results:
        run.write(r)
        run.write("\n")

    run.close()


def main():
    """The main program.

        Command line arguments, if provided:

        - args[0] contains the path to the index directory;
        - args[1] contains the path to the run file.
    """

    # Reads command-line arguments
    if len(sys.argv) == 2:
        index_path = sys.argv[1]
        run_file = sys.argv[2]
    else:
        index_path = DEFAULT_INDEX_PATH
        run_file = DEFAULT_RUN_FILE

    # Create the list of documents from the provided toy ones
    docs = parse_documents()

    # The analyzer to be used for document pre-processing
    # The StandardAnalyzer tokenizes documents using spaces and
    # recognizing, e.g., urls; then it turns tokens
    # lower-case and removes stop words.
    analyzer = StandardAnalyzer()

    # index documents
    index(index_path, docs, analyzer)

    # the list of results to be written to a run
    results = []

    # search the documents
    results = search(index_path, "001", "quokka", analyzer, results)
    results = search(index_path, "002", "Australia animals", analyzer, results)
    results = search(index_path, "003", "small marsupial", analyzer, results)
    results = search(index_path, "004", "quokka australia", analyzer, results)

    # print the results to a run file
    write_run_file(results, run_file)


if __name__ == "__main__":
    sys.exit(main())
