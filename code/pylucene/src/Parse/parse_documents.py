import io
import importlib
from abc import ABC, abstractmethod
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from langdetect import detect, LangDetectException
#import spacy
#from spacy.util import compile_infix_regex, compile_suffix_regex


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
        self.processor = TextProcessor()
        self.batch_size = 500  # Quanti documenti alla volta diamo in pasto a spaCy
        self.buffer = []
        '''
        for line in self._in:
            if line.strip(): # Skip empty lines
                self._data.append(json.loads(line))    
        self._iterator = iter(self._data)
        '''

    def _parse(self) -> ParsedDocument:
        # Se la scatola è vuota, carica i prossimi 500 documenti
        if not self.buffer:
            self._load_next_batch()
            
        # Se dopo aver provato a caricare la scatola è ancora vuota, il file è finito
        if not self.buffer:
            self._next = False
            raise StopIteration()
            
        # Tira fuori un documento dalla scatola e dallo a Lucene
        return self.buffer.pop(0)
    
    def _load_next_batch(self):
        raw_docs = []
        
        # 1. Legge le prossime 'batch_size' righe dal file JSONL
        for _ in range(self.batch_size):
            try:
                line = next(self._in)
                if line.strip():
                    item = json.loads(line)
                    doc_id = str(item.get("pubkey", ""))
                    title = item.get("title", "")
                    abstract = item.get("abstract", "")
                    body = f"{title}\n{abstract}".strip()
                    # Salviamo ID e testo originale
                    raw_docs.append((doc_id, body))
            except StopIteration:
                break # Abbiamo raggiunto la fine del file
                
        if not raw_docs:
            return

        # Find out the language of every doc
        docs_with_lang = []
        for doc_id, body in raw_docs:
            try:
                # detect() returns the language as 'en', 'fr', 'de', 'it'
                lang_code = detect(body)
                
                # Force the english if it's not in the previous 3 languages
                if lang_code not in ['en', 'fr', 'de']:
                    lang_code = 'en'
            except LangDetectException:
                lang_code = 'en'
                
            docs_with_lang.append((body, lang_code))
        
        # 3. Give to the processor list of tuples: [(testo1, 'en'), (testo2, 'fr'), ...]
        processed_texts = list(self.processor.process_batch(docs_with_lang))
        
        # 4. Rebuild all together
        for (doc_id, _), proc_body in zip(raw_docs, processed_texts):
            self.buffer.append(ParsedDocument(doc_id, proc_body))

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
# MULTILINGUAL TEXT PROCESSOR (NLTK-based)
# -----------------------------------------------------------------------------
# Handles English, French, and German text preprocessing.
# =============================================================================
class TextProcessor:
    def __init__(self):
        # We define the languages we want to support
        self.supported_langs = {
            "en": "english",
            "fr": "french",
            "de": "german"
        }
        
        self.stop_words = {}
        self.stemmers = {}
        
        # Pre-load stopwords and stemmers for all supported languages
        for short_code, lang_name in self.supported_langs.items():
            try:
                stopwords.words(lang_name)
            except LookupError:
                nltk.download('stopwords', quiet=True)
                
            self.stop_words[short_code] = set(stopwords.words(lang_name))
            self.stemmers[short_code] = SnowballStemmer(lang_name)
        
        # Fast regex tokenizer
        self.tokenizer = RegexpTokenizer(r'[A-Za-z0-9%\.\-]+')

    def process(self, text: str, lang_code: str = "en"):
        """
        Processes a single string using the correct language tools.
        Defaults to English ('en') if the language code is unknown.
        """
        # Fallback to English if the language is not fr or de
        if lang_code not in self.supported_langs:
            lang_code = "en"
            
        clean_text = re.sub(r'<[^>]+>', '', text).lower()
        tokens = self.tokenizer.tokenize(clean_text)
        
        processed_tokens = []
        for token in tokens:
            if len(token) > 1 and token not in self.stop_words[lang_code]:
                stemmed = self.stemmers[lang_code].stem(token)
                processed_tokens.append(stemmed)
                
        return processed_tokens

    def process_batch(self, docs_with_lang):
        """
        Expects a list of tuples: [(text1, lang1), (text2, lang2), ...]
        Yields processed strings.
        """
        for text, lang_code in docs_with_lang:
            yield " ".join(self.process(text, lang_code))

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