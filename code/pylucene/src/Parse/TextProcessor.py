import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextProcessor:
    def __init__(self):
        self.supported_langs = {
            "en": "english",
            "fr": "french",
            "de": "german"
        }
        
        self.stop_words = {}
        self.stemmers = {}
        
        for short_code, lang_name in self.supported_langs.items():
            self.stop_words[short_code] = set(stopwords.words(lang_name))
            self.stemmers[short_code] = SnowballStemmer(lang_name)
        
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*')

        self.medical_synonyms = {
            "covid": ["coronavirus", "sars-cov-2", "ncov", "2019-ncov", "covid-19", "covid19"],
            "vaccin": ["immunization", "shot", "jab", "vax", "inoculation"],
            "heart": ["cardiac", "cardiovascular", "myocarditis", "pericarditis"],
            "hcq": ["hydroxychloroquine", "chloroquine"],
            "mrna": ["messenger", "ribonucleic"],
            "zinc": ["zn", "gluconate"],
            "antiviral": ["anti-viral", "antimicrobial", "treatment"],
            "mask": ["respirator", "n95", "facemask", "face-mask"],
            "ivermectin": ["stromectol", "antiparasitic"]
        }

    def process(self, text: str, lang_code: str = "en"):
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

    def expand_query(self, text: str, lang_code: str = "en"):
        tokens = self.process(text, lang_code)
        expanded = set(tokens)

        for token in tokens:
            if token in self.medical_synonyms:
                for synonym in self.medical_synonyms[token]:
                    stemmed_synonym = self.stemmers["en"].stem(synonym)
                    expanded.add(stemmed_synonym)

        return list(expanded)

    def process_batch(self, docs_with_lang):
        for text, lang_code in docs_with_lang:
            yield " ".join(self.process(text, lang_code))