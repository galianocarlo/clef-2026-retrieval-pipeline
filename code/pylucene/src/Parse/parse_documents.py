import io
import importlib
from abc import ABC, abstractmethod
import json
import spacy
import re
from spacy.util import compile_infix_regex, compile_suffix_regex


##########################################################
# Building a parser ad hoc for the collection.jason data #
##########################################################
class ParsedDocument:
    # Simple data container representing a single document from the collection.
    # - id   : unique document identifier (mapped from the "pubkey" field)
    # - body : concatenation of title and abstract, used as the searchable text
    def __init__(self, doc_id: str, body: str):
        self.id = doc_id
        self.body = body

    def __str__(self):
        # Truncates body for readable debug output
        return f"ID: {self.id}\nBody: {self.body[:100]}...\n"

class DocumentParser:
    # Base parser class implementing the iterator protocol (__iter__ / __next__)
    # so that any concrete subclass can be used directly in a for-loop.
    def __init__(self, in_reader: io.TextIOBase):
        if in_reader is None:
            raise ValueError()
        self._next = True
        self._in = in_reader

    def __iter__(self):
        return self

    def __next__(self):
        if not self._next:
            raise StopIteration()
        try:
            return self._parse()
        finally:
            if not self._next:
                try:
                    self._in.close()
                except IOError as e:
                    raise RuntimeError() from e

    def _parse(self):
        # Base implementation: subclasses must override this method
        raise NotImplementedError()

class JsonArrayDocumentParser(DocumentParser):
    # Parser for JSONL files (JSON Lines format: one JSON object per line).
    
    def __init__(self, in_reader: io.TextIOBase):
        super().__init__(in_reader)
        self._data = []
        for line in self._in:
            if line.strip(): # Skip empty lines
                self._data.append(json.loads(line))    
        self._iterator = iter(self._data)

    def _parse(self) -> ParsedDocument:
        try:
            item = next(self._iterator)

            # Map JSONL fields to ParsedDocument attributes:
            # - "pubkey" is used as the unique document ID
            # - "title" and "abstract" are concatenated to form the document body
            doc_id = str(item.get("pubkey", ""))
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            
            # Combine title and abstract with a newline separator;
            # strip() removes any leading/trailing whitespace
            body = f"{title}\n{abstract}".strip()
            
            return ParsedDocument(doc_id, body)
            
        except StopIteration:
            # No more documents: signal the iterator protocol to stop
            self._next = False
            raise StopIteration()

# Quick test: prints the first 3 documents from the collection
if __name__ == "__main__":
    file_path = "experiment/docs/collection.jsonl" 
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            parser = JsonArrayDocumentParser(f)
            
            for i, doc in enumerate(parser):
                print(f"------------------------------------\n{doc}")
                if i == 2: 
                    break
    except FileNotFoundError:
        pass


# =============================================================================
# TEXT PROCESSOR
# -----------------------------------------------------------------------------
# Handles all NLP preprocessing steps needed before indexing or retrieval:
# HTML stripping, lowercasing, tokenization, stopword removal, and lemmatization.
# Uses spaCy's English model (en_core_web_sm).
# =============================================================================

class TextProcessor:
    def __init__(self, language="en"):
        model_name = "en_core_web_sm" 
        try:
            self.nlp = spacy.load(model_name)
        except OSError:        
            # If the model is not installed locally, download it automatically    
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        # Remove infix rules that split on hyphens ("-") to preserve
        # hyphenated terms as single tokens (e.g. "SARS-CoV-2" stays intact)
        infixes = self.nlp.Defaults.infixes
        custom_infixes = [rule for rule in infixes if "-" not in str(rule)]
        self.nlp.tokenizer.infix_finditer = compile_infix_regex(custom_infixes).finditer

        # Remove suffix rules that strip the "%" sign so that percentage values
        # like "99.9%" are kept as a single token rather than split
        suffixes = self.nlp.Defaults.suffixes
        custom_suffixes = [rule for rule in suffixes if "%" not in str(rule)]
        self.nlp.tokenizer.suffix_search = compile_suffix_regex(custom_suffixes).search

    def process(self, text: str):
        # Step 1: Strip HTML tags (e.g. <i>, <b>, <sup>) using a regex
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Step 2: Lowercase and run the full spaCy pipeline (tokenization, POS, etc.)
        doc = self.nlp(clean_text.lower())
        
        # Step 3: Filter tokens — keep only content words (no stopwords, punctuation, spaces)
        # and return their lemma (base form), e.g. "observed" -> "observe"
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                tokens.append(token.lemma_)
                
        return tokens


###########
# TESTING #
###########
if __name__ == "__main__":
    processor = TextProcessor(language="en")
    
    complex_text = "The observed efficacy <i>in vitro</i> for the SARS-CoV-2 virus is over 99.9%!!!"
    
    print("Original Text:")
    print(complex_text)
    print("-" * 40)
    
    result = processor.process(complex_text)
    
    print("Processed Tokens:")
    print(result)