
# this class is moved here to be saved from parse_document

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import re
from langdetect import detect, LangDetectException

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