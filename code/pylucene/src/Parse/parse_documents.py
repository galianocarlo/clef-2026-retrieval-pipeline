import io
from abc import ABC, abstractmethod
import json
import re
from collections import deque
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory # this is because langdetec is not deterministic

# To enforce consistent results
DetectorFactory.seed = 0

class ParsedDocument:
    """ As the professor, a python data container representing a single document from the collection.
     
     Attribute: 
        id: unique document identifier (mapped from the "pubkey" field)
        Title: title of the paper used as the searchable text
        Abstract: abstract used as the searchable text"""
    class FIELDS:    
        ID = "id"
        TITLE = "title"
        ABSTRACT = "abstract"
    
    def __init__(self, doc_id: str, title: str, abstract: str, lang_code: str = "en"):
        self.id = doc_id
        self.title = title
        self.abstract = abstract
        self.lang = lang_code

    def __str__(self):
        # Truncates body for readable debug output
        return f"ID: {self.id}\nTitle: {self.title}\nAbstract : {self.abstract[:100]}...\n"

class DocumentParser(ABC):
    """As the professor, abstract base class for parser implementing the python iterator protocol.
    
    Subclasses must implement _parse() to define how a single document is read from the input stream. 
    Doc: 
    - https://docs.python.org/3/tutorial/classes.html#iterators"""

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
        """Read and return the next document from the input stream"""
        pass

class JsonArrayDocumentParser(DocumentParser):
    """
    Parser for JSONL files (JSON Lines format: one JSON object per line).
    """
    def __init__(self, in_reader: io.TextIOBase):
        super().__init__(in_reader)
        self.batch_size = 500  # number of lines we take
        self.buffer = deque()

    def _parse(self) -> ParsedDocument:
        # If buffer is empty, load next up to 500 docs
        if not self.buffer:
            self._load_next_batch()
            
        # if the buffer is still empty, meaning the files is closed
        if not self.buffer:
            self._next = False
            raise StopIteration()
            
        # it returns a doc to pass it to Lucene
        return self.buffer.popleft()
    
    def _load_next_batch(self):
        raw_docs = []
        
        for _ in range(self.batch_size):
            try:
                line = next(self._in)
                if line.strip():
                    item = json.loads(line)
                    doc_id = str(item.get("pubkey", ""))
                    title = item.get("title", "")
                    abstract = item.get("abstract", "")
                    
                    # Collect the id, title and abstract in our list
                    raw_docs.append((doc_id, title, abstract))
            except StopIteration:
                break # end of the files
                
        if not raw_docs:
            return

        for doc_id, title, abstract in raw_docs:
            try:
                # detect() returns the language as 'en', 'fr', 'de'
                lang_code = detect(f"{title} {abstract}")
                
                # Force the english if it's not in the previous 3 languages
                if lang_code not in ['en', 'fr', 'de']:
                    lang_code = 'other'
            except LangDetectException:
                lang_code = 'other'
            
            self.buffer.append(ParsedDocument(doc_id, title, abstract, lang_code))


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


