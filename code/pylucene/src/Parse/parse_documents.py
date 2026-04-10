import io
import importlib
from abc import ABC, abstractmethod
import json
import re
from langdetect import detect, LangDetectException
#from src.Parse.TextProcessor import TextProcessor

# TO DO
# Update ParsedDocument.FIELDS to include ABSTRACT, AUTHOR, VENUE, and LANG.
# Update the ParsedDocument.__init__ to accept those new fields.
# Update the _load_next_batch method to extract those fields from the item dictionary and pass them to ParsedDocument().
# 

class ParsedDocument:
    # Single Source of Truth
    class FIELDS:
        ID = "id"
        TITLE = "title"
        ABSTRACT = "abstract"
    # Simple data container representing a single document from the collection.
    # - id   : unique document identifier (mapped from the "pubkey" field)
    # - body : concatenation of title and abstract, used as the searchable text
    def __init__(self, doc_id: str, title: str, abstract: str, lang_code: str = "en"):
        self.id = doc_id
        self.title = title
        self.abstract = abstract
        self.lang = lang_code
        # we will divide body into each field (abstract, authours and more important lang (we could use it for improving the indexer and searcher))

    def __str__(self):
        # Truncates body for readable debug output
        return f"ID: {self.id}\nTitle: {self.title}\nAbstract : {self.abstract[:100]}...\n"

class DocumentParser(ABC):
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
    @abstractmethod
    def _parse(self):
        # Base implementation: subclasses must override this method
        pass

class JsonArrayDocumentParser(DocumentParser):
    # Parser for JSONL files (JSON Lines format: one JSON object per line).
    
    def __init__(self, in_reader: io.TextIOBase):
        super().__init__(in_reader)
        #self.processor = TextProcessor()
        self.batch_size = 500  # number of lines we take
        self.buffer = []

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
                    
                    # Salviamo ID, titolo e contenuto
                    raw_docs.append((doc_id, title, abstract))
            except StopIteration:
                break # Abbiamo raggiunto la fine del file
                
        if not raw_docs:
            return

        # Find out the language of every doc
        docs_with_lang_title = []
        docs_with_lang_abstract = []
        
        for doc_id, title, abstract in raw_docs:
            try:
                # detect() returns the language as 'en', 'fr', 'de', 'it'
                lang_code = detect(f"{title} {abstract}")
                
                # Force the english if it's not in the previous 3 languages
                if lang_code not in ['en', 'fr', 'de']:
                    lang_code = 'other'
            except LangDetectException:
                lang_code = 'other'
            
            self.buffer.append(ParsedDocument(doc_id, title, abstract, lang_code))
            '''    
            docs_with_lang_title.append((title, lang_code))
            docs_with_lang_abstract.append((abstract, lang_code))
        
        # 3. Give to the processor list of tuples: [(testo1, 'en'), (testo2, 'fr'), ...]
        processed_titles = list(self.processor.process_batch(docs_with_lang_title))
        processed_abstracts = list(self.processor.process_batch(docs_with_lang_abstract))
        
        # 4. Rebuild all together
        for (doc_id, _, _), (_, lang_code), proc_title, proc_abstract in zip(raw_docs, docs_with_lang_title, processed_titles, processed_abstracts):
            self.buffer.append(ParsedDocument(doc_id, proc_title, proc_abstract, lang_code))
        '''    
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


