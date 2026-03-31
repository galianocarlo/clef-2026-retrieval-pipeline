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
    def __init__(self, doc_id: str, title: str, abstact: str):
        self.id = doc_id
        self.title = title
        self.abstract = abstact

    def __str__(self):
        # Truncates body for readable debug output
        return f"ID: {self.id}\nTitle: {self.title}\nAbstract: {self.abstract[:100]}...\n"

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
           
            doc_id = str(item.get("pubkey", ""))
            title = item.get("title", "")
            abstract = item.get("abstract", "")
              

            return ParsedDocument(doc_id, title, abstract)
            
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
        models = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
            "es": "es_core_news_sm"
        }
        
        model_name = models.get(language, "en_core_web_sm")
        
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

if __name__ == "__main__":
    processor_en = TextProcessor(language="en")
    processor_fr = TextProcessor(language="fr")
    processor_de = TextProcessor(language="de")
    processor_it = TextProcessor(language="it")
    processor_es = TextProcessor(language="es")
    
    text_en = "The observed efficacy <i>in vitro</i> for the SARS-CoV-2 virus is over 99.9%!!!"
    text_fr = "L'efficacité observée <i>in vitro</i> pour le virus SARS-CoV-2 est supérieure à 99,9%!!!"
    text_de = "Die beobachtete Wirksamkeit <i>in vitro</i> für das SARS-CoV-2-Virus liegt bei über 99,9%!!!"
    text_it = "L'efficacia osservata <i>in vitro</i> per il virus SARS-CoV-2 è superiore al 99,9%!!!"
    text_es = "¡La eficacia observada <i>in vitro</i> para el virus SARS-CoV-2 es superior al 99,9%!!!"
    
    print(processor_en.process(text_en))
    print(processor_fr.process(text_fr))
    print(processor_de.process(text_de))
    print(processor_it.process(text_it))
    print(processor_es.process(text_es))