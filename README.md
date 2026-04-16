# Search Engines (SE) - Repository Template

This repository is a template repository for the homeworks to be developed in the Search Engines course. A.Y. 2025/2026.

The homeworks are carried out by groups of students and consists in participating to one of the labs organized yearly by [CLEF](https://www.clef-initiative.eu/) (Conference and Labs of the Evaluation Forum).

*Search Engines* is a course of the

* [Master Degree in Computer Engineering](https://degrees.dei.unipd.it/master-degrees/computer-engineering/) of the  [Department of Information Engineering](https://www.dei.unipd.it/en/), [University of Padua](https://www.unipd.it/en/), Italy.
* [Master Degree in Data Science](https://datascience.math.unipd.it/) of the  [Department of Mathematics "Tullio Levi-Civita"](https://www.math.unipd.it/en/), [University of Padua](https://www.unipd.it/en/), Italy.

*Search Engines* is part of the teaching activities of the [Intelligent Interactive Information Access (IIIA) Hub](http://iiia.dei.unipd.it/).

### Team Members ###

- Stefano Riato (stefano.riato@studenti.unipd.it)
- Alberto Iurlo (alberto.iurlo@studenti.unipd.it)
- Carlo Galiano (carlo.galiano.1@studenti.unipd.it)
- Nathasha Nedumkulam (nathasha.nedumkulam@studenti.unipd.it)


### Organisation of the repository ###

The repository is organised as follows:

* `code`: this folder contains the source code of the developed system. See dedicated section below for more details.
* `runs`: this folder contains the runs produced by the developed system.
* `results`: this folder contains the performance scores of the runs.
* `homework-1`: this folder contains the report describing the techniques applied and insights gained.
* `homework-2`: this folder contains the final paper submitted to CLEF.
* `slides`: this folder contains the slides used for presenting the conducted project.

### License ###

All the contents of this repository are shared using the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

![CC logo](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)



## Organization of the `code` folder

The `code` folder serves as starter for the participation at [CheckThat! 2026](https://checkthat.gitlab.io/clef2026/task1/) at [CLEF 2026](https://clef2026.clef-initiative.eu/).

This section contains a multi-stage search pipeline for medical documents, evolving from a standard keyword-based search to a sophisticated neural-hybrid system.

### 1. `experiment/runs/` (Output Files)

This directory contains the results of different search configurations. Each file represents an evolutionary step in our pipeline:

* **`seupd-baseline-run.txt`**: The initial baseline run used as a reference point for comparisons.
* **`run-native-analyzers.txt`**: Base search results (V1) using PyLucene with BM25 and native multilingual analyzers.
* **`run-native-analyzers-prf.txt`**: Advanced search results (V2). This implements BM25 enhanced with Pseudo-Relevance Feedback (automatic medical term expansion).
* **`run-neural-reranked.txt`**: Results of Neural Re-ranking applied to the base run (V1). A cross-encoder model re-orders the top results based on semantic meaning.
* **`run-neural-reranked-prf.txt`**: Results of Neural Re-ranking applied to the expanded PRF run (V2). This is our most powerful semantic evaluation.
* **`run-hybrid-best.txt`**: Best merged run (V3). A "Frankenstein" pipeline combining precise BM25 lexical matching for English with neural semantic understanding for German and French.
* **`run-hybrid-best-prf.txt`**: "V3 + PRF (Experimental): Hybrid pipeline built on the PRF expansion, combining BM25+PRF for English with Neural+PRF re-ranking for DE and FR."
* **`run-hybrid-reranked.txt`**: "V4 (Asymmetric Hybrid): Language-routed pipeline using pure lexical BM25 for English queries, and a Dense+Lexical Hybrid approach with Neural Re-ranking strictly for German and French."

### 2. `pylucene/src/Evaluation/` (Scoring & Merging)

* **`evaluate.py`**: Computes official metrics (MRR, MAP, NDCG) by comparing our generated runs against the ground truth.
* **`generate_qrels.py`**: Formats the "gold standard" or ground truth (qrels) files required for evaluation.
* **`merge_runs.py`**: A utility script that creates the final hybrid run by combining English results from BM25 with DE/FR results from the Neural Re-ranker.

### 3. `pylucene/src/Index/` (Indexing)

* **`index.py`**: The core module that reads the collection of scientific papers and builds the PyLucene inverted index.

### 4. `pylucene/src/Parse/` (Data Pre-processing)

* **`parse_documents.py`**: Handles reading, cleaning, and structuring raw medical papers for indexing.
* **`parse_query.py`**: Processes user tweets, formatting queries and extracting entities (such as medical journals using `journals.txt`).
* **`TextProcessor.py`**: General utility functions for string processing, including cleaning and basic normalization.

### 5. `pylucene/src/Search/` (Search Engines)

* **`search.py`**: The primary search engine. It utilizes PyLucene and BM25 to perform asymmetric multilingual lexical searches.
* **`search_prf.py`**: An evolution of the primary search engine. It implements a two-stage retrieval process using Pseudo-Relevance Feedback to expand queries.
* **`rerank.py`**: The AI engine. It uses a HuggingFace multilingual Cross-Encoder to evaluate document semantics and re-rank candidate results.
