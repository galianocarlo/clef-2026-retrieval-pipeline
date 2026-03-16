import sys
import os
import json

from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

# The first toy document.
# Credits to Maria Maistro.
D1 = "The quokka, the only member of the genus Setonix, is a small marsupial about the size of a domestic cat"

# The second toy document.
# Credits to Maria Maistro.
D2 = "Wombats are small, short-legged, muscular quadrupedal marsupials that are native to Australia"

# The third toy document.
# Credits to Maria Maistro.
D3 = "Quokkas have little fear of humans and commonly approach people closely, particularly on Rottnest Island in Australia, where a prevalent population exists"

# The name of the id field in a Document.
ID = "id"

# The name of the body field in a Document.
BODY = "contents"  # Note: Pyserini uses "contents" as the default searchable field

# The default path to the directory containing the raw documents (jsonl)
DEFAULT_DOCS_PATH = "experiment/docs"

# The default path to the directory containing the index
DEFAULT_INDEX_PATH = "experiment/index/pyserini"

# The default run file
DEFAULT_RUN_FILE = "experiment/runs/run-pyserini.txt"

# The identifier of the run to be used in the run file.
RUN_ID = "hello-IR-pyserini"

# The maximum number of documents to be retrieved for each query
MAX_DOCS_RETRIEVED = 10


def parse_documents():
    """Parses the toy documents and returns a list of dicts in Pyserini format.

    Pyserini expects documents as JSON objects with at minimum:
    - "id":       a unique string identifier
    - "contents": the text to be indexed and searched

    Returns:
        list: a list of dicts representing the documents
    """

    print("------------- PARSING DOCUMENTS -------------")

    # Pyserini works with plain Python dicts instead of Java Document objects
    docs = [
        {ID: "d1", BODY: D1},
        {ID: "d2", BODY: D2},
        {ID: "d3", BODY: D3},
    ]

    print("The documents are:")
    for d in docs:
        print(d)

    print("---------------------------------------------\n")

    return docs


def index(docs_path, index_path, docs):
    """Writes documents to disk as JSONL and indexes them with Pyserini.

    Pyserini's LuceneIndexer requires documents to be saved as .jsonl files
    on disk before indexing. This is different from PyLucene where documents
    are passed directly in memory.

    Args:
        docs_path (str): The path where to store the raw .jsonl documents.
        index_path (str): The path where to store the index.
        docs (list): a list of document dicts to be indexed.
    """

    print("------------- INDEXING DOCUMENTS -------------")

    # Create the docs folder if it does not exist
    os.makedirs(docs_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)

    # Write documents to a .jsonl file (one JSON object per line)
    # This is the format Pyserini expects as input for indexing
    jsonl_path = os.path.join(docs_path, "docs.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    print(f"- Documents written to {jsonl_path}")

    # Index using LuceneIndexer with BM25 (the default similarity in Pyserini,
    # matching the BM25Similarity used in the PyLucene version)
    # Note: Depending on your Pyserini version, you may need to pass args here.
    indexer = LuceneIndexer(index_path, append=False)

    for d in docs:
        indexer.add_doc_dict(d)
        print(f"- Document {d[ID]} successfully indexed")

    indexer.close()

    print("- Indexer successfully closed")
    print("----------------------------------------------\n")


def search(index_path, query_id, query, results):
    """Searches for the provided query using BM25.

    Args:
        index_path (str): The path to the index directory.
        query_id (str): The id of the query to search for.
        query (str): The query text to search for.
        results (list): a list of results in trec_eval format.

    Returns:
        list: the updated list of results in trec_eval format.
    """

    print(f"------------- SEARCHING DOCUMENTS FOR QUERY {query_id} -------------")

    # Open the index with a LuceneSearcher
    # This is the Pyserini equivalent of opening FSDirectory + IndexSearcher
    searcher = LuceneSearcher(index_path)

    # Set BM25 similarity explicitly to match the PyLucene version
    # Default k1=0.9, b=0.4 are Pyserini's defaults (Lucene defaults)
    searcher.set_bm25()

    print("- Searcher successfully created")
    print(f"- Query: {query}")

    # Perform the actual search — equivalent to searcher.search(q, MAX_DOCS_RETRIEVED)
    hits = searcher.search(query, k=MAX_DOCS_RETRIEVED)

    print(f"- {len(hits)} document(s) retrieved")

    for i, hit in enumerate(hits):
        # hit.docid  -> equivalent to doc.get(ID) in PyLucene
        # hit.score  -> equivalent to hits[i].score in PyLucene
        result_line = f"{query_id}\tQ0\t{hit.docid}\t{i}\t{hit.score:.6f}\t{RUN_ID}"
        results.append(result_line)
        print(result_line)

    print("- Searcher successfully closed")
    print("-------------------------------------------------------------\n")

    return results


def write_run_file(results, run_file):
    """Writes the results to a run file in trec_eval format.

    Args:
        results (list): a list of results for one or more queries.
        run_file (str): the path to the run file.
    """

    # Create the output folder if it does not exist
    os.makedirs(os.path.dirname(run_file), exist_ok=True)

    with open(run_file, "w", encoding="utf-8") as run:
        for r in results:
            run.write(r + "\n")

    print(f"Run file written to: {run_file}")


def main():
    """The main program.

    Command line arguments, if provided:

    - args[1] contains the path to the index directory;
    - args[2] contains the path to the run file.
    """

    # Reads command-line arguments
    if len(sys.argv) == 3:
        index_path = sys.argv[1]
        run_file = sys.argv[2]
    else:
        index_path = DEFAULT_INDEX_PATH
        run_file = DEFAULT_RUN_FILE

    docs_path = DEFAULT_DOCS_PATH

    # Create the list of documents from the provided toy ones
    docs = parse_documents()

    # Index documents
    # Note: Pyserini handles the analyzer internally (StandardAnalyzer by default),
    # so unlike PyLucene we do not need to instantiate it explicitly
    index(docs_path, index_path, docs)

    # The list of results to be written to a run
    results = []

    # Search the documents — same queries as the PyLucene version
    results = search(index_path, "001", "quokka", results)
    results = search(index_path, "002", "Australia animals", results)
    results = search(index_path, "003", "small marsupial", results)
    results = search(index_path, "004", "quokka australia", results)

    # Write results to a run file
    write_run_file(results, run_file)


if __name__ == "__main__":
    sys.exit(main())