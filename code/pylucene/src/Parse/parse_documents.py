import io
import importlib
from abc import ABC, abstractmethod
import json
import SpaCy

class ParsedDocument:
    pass

class DocumentParser(ABC):

    def __init__(self, in_reader: io.TextIOBase):
        if in_reader is None:
            raise ValueError("Reader cannot be null.")
        
        self._next = True
        self._in = in_reader

    def __iter__(self):
        return self

    def __next__(self) -> ParsedDocument:
        if not self._next:
            raise StopIteration()

        try:
            return self._parse()
        finally:
            if not self._next:
                try:
                    self._in.close()
                except IOError as e:
                    raise RuntimeError("Unable to close the reader.") from e

    @staticmethod
    def create_from_class(cls_type, in_reader: io.TextIOBase):
        if cls_type is None:
            raise ValueError("Document parser class cannot be null.")
        if in_reader is None:
            raise ValueError("Reader cannot be null.")
        
        try:
            return cls_type(in_reader)
        except Exception as e:
            raise RuntimeError(f"Unable to instantiate document parser {cls_type.__name__}.") from e

    @staticmethod
    def create_from_string(cls_name: str, in_reader: io.TextIOBase):
        if not cls_name or not cls_name.strip():
            raise ValueError("Document parser class cannot be null or empty.")
        
        try:
            module_path, class_name_str = cls_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls_type = getattr(module, class_name_str)
            return DocumentParser.create_from_class(cls_type, in_reader)
        except Exception as e:
            raise RuntimeError(f"Unable to find the class of the document parser {cls_name}.") from e

    @abstractmethod
    def _parse(self) -> ParsedDocument:
        pass


#########################################################
#Building a parser ad hoc for the collection.jason data #
#########################################################
class ParsedDocument:
    def __init__(self, doc_id: str, body: str):
        self.id = doc_id
        self.body = body

    def __str__(self):
        return f"ID: {self.id}\nBody: {self.body[:100]}...\n"

class DocumentParser:
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
        raise NotImplementedError()

class JsonArrayDocumentParser(DocumentParser):
    def __init__(self, in_reader: io.TextIOBase):
        super().__init__(in_reader)
        self._data = []
        for line in self._in:
            if line.strip():
                self._data.append(json.loads(line))    
        self._iterator = iter(self._data)

    def _parse(self) -> ParsedDocument:
        try:
            item = next(self._iterator)
            
            doc_id = str(item.get("pubkey", ""))
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            
            body = f"{title}\n{abstract}".strip()
            
            return ParsedDocument(doc_id, body)
            
        except StopIteration:
            self._next = False
            raise StopIteration()

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


#################
#Text processing#
#################

import spacy
import re
from spacy.util import compile_infix_regex, compile_suffix_regex

class TextProcessor:
    def __init__(self, language="en"):
        model_name = "en_core_web_sm" 
        try:
            self.nlp = spacy.load(model_name)
        except OSError:            
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        infixes = self.nlp.Defaults.infixes
        custom_infixes = [rule for rule in infixes if "-" not in str(rule)]
        self.nlp.tokenizer.infix_finditer = compile_infix_regex(custom_infixes).finditer

        suffixes = self.nlp.Defaults.suffixes
        custom_suffixes = [rule for rule in suffixes if "%" not in str(rule)]
        self.nlp.tokenizer.suffix_search = compile_suffix_regex(custom_suffixes).search

    def process(self, text: str):
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        doc = self.nlp(clean_text.lower())
        
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