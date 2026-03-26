# Introduction to Apache Lucene: Architecture and Workflow

If you are accustomed to working with traditional relational databases (like SQL) or flat files (like CSV/Excel), Apache Lucene (and its Python wrapper, PyLucene) requires a complete mental shift. 

Traditional databases scan row-by-row to find a match. Lucene does the exact opposite: it builds an **Inverted Index**—much like the index at the back of a massive textbook. Instead of asking the database, *"Which words are in Document #4?"*, Lucene's index instantly answers, *"Which documents contain the word 'vaccine'?"*

*(Source: [Official Apache Lucene Documentation](https://lucene.apache.org/core/7_3_1/core/overview-summary.html#overview.description))*

---

## The Standard Lucene Workflow
To successfully implement Lucene in an application, you generally follow a four-step process:

1. **Populate the Data:** Create `Document` objects and populate them with `Field` objects.
2. **Build the Index:** Create an `IndexWriter` and feed your documents into it using the `addDocument()` method.
3. **Parse the User's Request:** Use `QueryParser.parse()` to translate a standard search string into a structured Lucene query.
4. **Retrieve Results:** Create an `IndexSearcher` and pass your parsed query to its `search()` method to retrieve the top matching documents.

---

## The Core API Packages
To achieve this highly optimized search process, the Lucene API is divided into several specialized packages. Here is a breakdown of the core architecture:

* **`org.apache.lucene.analysis` (Text Processing)**
  Defines the abstract `Analyzer` API used to convert raw text into a `TokenStream` (an enumeration of token attributes). A `TokenStream` is built by applying `TokenFilters` to the output of a `Tokenizer`. The `analyzers-common` library provides several ready-to-use implementations, such as the `StopAnalyzer` and the grammar-based `StandardAnalyzer`.

* **`org.apache.lucene.document` (Data Containers)**
  Provides the fundamental `Document` class. A `Document` acts as a simple container representing a single search result. It is composed of a set of named `Fields`, whose values can be strings or text readers.

* **`org.apache.lucene.index` (Index Management)**
  Contains the core indexing mechanisms. It provides two primary classes: the `IndexWriter`, which builds the index and adds documents to it, and the `IndexReader`, which accesses and reads the compiled data.

* **`org.apache.lucene.store` (Storage Layer)**
  Defines an abstract `Directory` class for storing persistent data, essentially acting as a collection of named files written by an `IndexOutput` and read by an `IndexInput`. It includes multiple implementations based on your application's needs, such as `FSDirectory` (for storing files directly on a hard drive) and `RAMDirectory` (for keeping files in active memory).

* **`org.apache.lucene.search` (Query & Retrieval)**
  Provides the data structures needed to represent user queries (e.g., `TermQuery` for single words, `PhraseQuery` for exact phrases, and `BooleanQuery` for complex logical combinations). It includes the `IndexSearcher`, which evaluates queries and returns `TopDocs` (ranked results), as well as various `QueryParsers` for translating text strings or XML into these structured queries.

* **`org.apache.lucene.codecs` (Internal Encoding)**
  Provides an abstraction layer over the actual encoding and decoding of the inverted index structure on the disk. It offers different implementations that can be swapped depending on specific application requirements.

* **`org.apache.lucene.util` (Background Tools)**
  Contains highly optimized data structures and utility classes used by the engine under the hood, such as `FixedBitSet` and `PriorityQueue`.

---

## 1. `Document` (The Manila Folder)
In Lucene, a `Document` is the atomic unit of search. 
* **The Analogy:** Think of it as an empty manila folder. 
* **How it works:** A `Document` has no fixed schema. Unlike an SQL table that forces every row to have the same columns, a Lucene `Document` is just a flexible container. You can put one piece of information in Document A, and fifty pieces in Document B. 

## 2. `Field` (The Papers Inside)
If the `Document` is the folder, the `Field` is the actual piece of paper you put inside it. A `Document` is made up of multiple `Fields` (e.g., an "ID" field, a "Title" field, a "Body" field).
* **The Analogy:** The labeled sheets of paper inside your manila folder.
* **How it works:** Lucene searches *Fields*, not overall Documents. When a user searches, they are technically saying, *"Find me the word 'COVID' specifically inside the 'Body' field."*

## 3. `FieldType` & `IndexOptions` (The Filing Instructions)
This is the most critical part of our memory optimization. When you hand a `Field` to Lucene, you attach a `FieldType` ([see documentation](https://lucene.apache.org/core/7_3_1/core/org/apache/lucene/document/FieldType.html)). This acts as an instruction manual for the database engine.

In our code, we configured our `BodyField` with very specific rules to keep our index small and fast:
* **`setTokenized(True)`:** We tell Lucene to chop the sentences into individual words (tokens).
* **`setIndexOptions(IndexOptions.DOCS_AND_FREQS)`:** We tell Lucene to record *which* documents a word appears in (DOCS), and *how many times* it appears (FREQS) for our BM25 scoring. We explicitly skip saving the exact position of the words to save disk space.
* **`setStored(False)`:** **This is crucial.** We tell Lucene *not* to save the original, readable text in the database. It shreds the original paper after counting the words. We can *search* the text, but we cannot ask the index to print the original paragraph back to us. 
* *Note: We set `Field.Store.YES` for our "ID" field so we can retrieve the ID and look up the original text in our raw `.jsonl` files later.*

## 4. `Analyzer` (The Language Expert)
Before a word is written into the database, it passes through an `Analyzer`.
* **The Analogy:** The editor who cleans up the text before filing it.
* **How it works:** We use the `StandardAnalyzer`. If our raw text is *"The researchers are testing!"*, the Analyzer lowercases everything, strips the punctuation, and removes "stop words" (like "the", "are"). The database only ends up storing *"researchers"* and *"testing"*.

## 5. `IndexWriter` (The Factory Manager)
The `IndexWriter` is the engine that executes our instructions.
* **The Analogy:** The factory manager who oversees the whole assembly line.
* **How it works:** We feed our filled `Document` folders to the `IndexWriter`. It applies the `Analyzer`, follows the `FieldType` rules, and physically writes the highly optimized Inverted Index to our hard drive (using `FSDirectory`). 



---

### Summary of Our Pipeline Flow
1. Our custom Python parser (`CollectionParser`) reads the raw `.jsonl` files line-by-line.
2. We create an empty Lucene `Document`.
3. We attach a stored `StringField` for the Document ID.
4. We attach an un-stored, tokenized `Field` for the main text Body.
5. The `IndexWriter` processes the Document and updates the Inverted Index on disk.