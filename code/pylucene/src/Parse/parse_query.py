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
                 pubkey: str, journal_list: list):
        self.query_id     = query_id
        self.raw_text     = text
        self.language     = language
        self.pubkey       = pubkey
        self.journal_list = journal_list

    def get_search_query(self) -> str:
        text = self.raw_text
        text = re.sub(r'@\w+|http\S+', '', text)
        text = text.replace("#", "")
        allowed = ".,!?;-()" # now keep the hyphen (and percentage % as in the paper(?))
        text = "".join(c for c in text if c.isalnum() or c.isspace() or c in allowed)
        text = re.sub(r'^\d+[\)/]\s*', '', text)
        return " ".join(text.split()).strip()

    def extract_entities(self) -> list:
        return [
            j for j in self.journal_list
            if re.search(r'\b' + re.escape(j) + r'\b', self.raw_text.lower())
        ]

    def __str__(self):
        entities = self.extract_entities()
        tag = f" [JOURNALS: {entities}]" if entities else ""
        return (f"--- Query #{self.query_id} ({self.language.upper()}) ---"
                f"{tag}\n"
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

    def __init__(self, in_reader: io.TextIOBase, lang_code: str, journal_list: list):
        super().__init__(in_reader)
        self.lang_code    = lang_code
        self.journal_list = journal_list
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
                        journal_list = self.journal_list,
                    ))
                    count += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line.")


def load_journals(filepath: str) -> list:
    """Load journal names from a plain-text file, one per line."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"journals file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


if __name__ == "__main__":
    SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    JOURNALS_PATH = os.path.join(SCRIPT_DIR, "journals.txt")
    file_path     = "experiment/docs/en_train.jsonl"
    lang          = 'en'

    journals = load_journals(JOURNALS_PATH)  # ← load first
    print(journals)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            parser = JsonArrayQueryParser(f, lang, journals)  # ← pass list, not path
            for i, doc in enumerate(parser):
                print(f"------------------------------------\n{doc}")
                if i == 10:
                    break
    except FileNotFoundError:
        pass
