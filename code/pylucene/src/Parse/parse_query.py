import io
import json
import os 
import re
from abc import ABC, abstractmethod
from collections import deque

class ParsedQuery:
    """As the professor, a python data container representing a single query from the topics.
     
     Attribute: 
        id: unique query identifier
        Text: the string of the query
        Language: language used in the query
        pubkey: the single paper the query refers to"""
    class FIELDS:
        ID       = "query_id"
        TEXT     = "text"
        LANGUAGE = "language"
        PUBKEY   = "pubkey"

    def __init__(self, query_id, text: str, language: str,
                 pubkey: str):
        self.query_id     = query_id
        self.raw_text     = text
        self.language     = language
        self.pubkey       = pubkey

    def get_search_query(self) -> str:
        text = self.raw_text
        text = re.sub(r'@\w+|http\S+', '', text)
        text = text.replace("#", "")
        allowed = ".,!?;-()" # now keep the hyphen (and percentage % as in the paper(?))
        text = "".join(c for c in text if c.isalnum() or c.isspace() or c in allowed)
        text = re.sub(r'^\d+[\)/]\s*', '', text)
        # Lucene treats uppercase AND / OR / NOT as boolean operators; lowercase them
        # so a tweet starting with "AND THIS HAPPENED..." doesn't crash the parser.
        text = re.sub(r'\b(AND|OR|NOT)\b', lambda m: m.group().lower(), text)
        return " ".join(text.split()).strip()

    def __str__(self):
        return (f"--- Query #{self.query_id} ({self.language.upper()}) ---"
                f"   CLEANED: {self.get_search_query()[:200]}...\n")
    

class QueryParser(ABC):
    """As the professor, abstract base class for parser implementing the python iterator protocol.
    
    Subclasses must implement _parse() to define how a single query is read from the input stream"""
    def __init__(self, in_reader: io.TextIOBase):
        if in_reader is None:
            raise ValueError("in_reader cannot be None.")
        self._next = True
        self._in   = in_reader

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
    def _parse(self) -> ParsedQuery:
        pass


class JsonArrayQueryParser(QueryParser):
    """Parser for JSONL query files """

    def __init__(self, in_reader: io.TextIOBase, lang_code: str):
        super().__init__(in_reader)
        self.lang_code    = lang_code
        self.batch_size   = 500
        self.buffer       = deque()

    def _parse(self) -> ParsedQuery:
        if not self.buffer:
            self._load_next_batch()
        if not self.buffer:
            self._next = False
            raise StopIteration()
        return self.buffer.popleft()

    def _load_next_batch(self):
        count = 0
        while count < self.batch_size:
            try:
                line = next(self._in)
            except StopIteration:
                break
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if "index" in entry and "text" in entry:
                    self.buffer.append(ParsedQuery(
                        query_id     = entry["index"],
                        text         = entry["text"],
                        language     = self.lang_code,
                        pubkey       = entry.get("pubkey"),
                    ))
                    count += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line.")


if __name__ == "__main__":
    SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    file_path     = "experiment/docs/en_train.jsonl"
    lang          = 'en'

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            parser = JsonArrayQueryParser(f, lang)  # ← pass list, not path
            for i, doc in enumerate(parser):
                print(f"------------------------------------\n{doc}")
                if i == 10:
                    break
    except FileNotFoundError:
        pass
