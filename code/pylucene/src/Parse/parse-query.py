import json
import os 
import re

class ScientificClaimQuery:
    """
    Represents a scientific claim extracted from social media.
    """
    def __init__(self, query_id, text, language, pubkey, journal_list):
        self.query_id = query_id
        self.raw_text = text 
        self.language = language
        self.pubkey = pubkey
        # We pass the loaded journal list into each query object
        self.journal_list = journal_list

    def get_search_query(self):
        text = self.raw_text
        
        # 1. Standard cleaning (Mentions and URLs)
        text = re.sub(r'@\w+|http\S+', '', text)
        text = text.replace("#", "")

        # 2. SMART FILTER: Keep letters (including German/French), numbers, 
        # spaces, and basic punctuation. This automatically drops Emojis.
        allowed_punctuation = ".,!?;-()/"
        text = "".join(char for char in text if char.isalnum() or char.isspace() or char in allowed_punctuation)

        # 3. Protective Thread Cleaning (Start of line only)
        text = re.sub(r'^\d+[\)/]\s*', '', text)
        
        # 4. Final Polish
        return " ".join(text.split()).strip()

    def extract_entities(self):
        """
        Checks the text against the journal list loaded from journals.txt.
        """
        # We look for each journal name from our list in the raw text
        found = [j for j in self.journal_list if j in self.raw_text.lower()]
        return found

    def __str__(self):
        clean_text = self.get_search_query()
        entities = self.extract_entities()
        tag = f" [JOURNALS: {entities}]" if entities else ""
        
        return (f"--- Claim #{self.query_id} ({self.language.upper()}) ---"
                f"{tag}\n"
                f"   CLEANED: {clean_text[:200]}...\n")

class ClaimParser:
    # Now it takes TWO specific paths
    def __init__(self, data_dir, journals_file):
        self.data_dir = data_dir
        # Pass the full path directly to the loader
        self.journals = self.load_journals(journals_file)

    def load_journals(self, filepath):
        """
        Reads the journals.txt file using the exact filepath provided.
        """
        # We don't need os.path.join here because 'filepath' is already the exact GPS coordinate
        if not os.path.exists(filepath):
            print(f"!!! Warning: {filepath} not found. Entity extraction will be empty.")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]

    def parse_language_file(self, filename, lang_code):
        # We use the new variable name 'self.data_dir' here!
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"!!! Error: {file_path} not found.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
               
                for line in f:
                    if not line.strip(): continue # Skip blank lines
                    try:
                        entry = json.loads(line) # Notice the 's' in loads()
                        if "index" in entry and "text" in entry:
                            yield ScientificClaimQuery(
                                query_id=entry["index"],
                                text=entry["text"],
                                language=lang_code,
                                pubkey=entry.get("pubkey"),
                                journal_list=self.journals 
                            )
                    except json.JSONDecodeError:
                        print(f"!!! Error: {filename} contains invalid JSON.")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Finds the exact folder this script is sitting in (.../pylucene/src/Parse)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Points directly to the journals.txt sitting next to it
    JOURNALS_PATH = os.path.join(SCRIPT_DIR, "journals.txt")
    
    # 3. Navigates UP three folders, then DOWN into experiment/docs
    DATA_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "experiment", "docs"))

    # 4. Gives the robot the exact GPS coordinates to both locations
    parser = ClaimParser(data_dir=DATA_FOLDER, journals_file=JOURNALS_PATH)
    
    tasks = [
        ("en_dev.jsonl", "en"),
        ("de_dev.jsonl", "de"),
        ("fr_dev.jsonl", "fr")
    ]

    all_claims = []

    for file_name, lang in tasks:
        print(f"\n>>> Loading {lang.upper()} collection...")
        for claim in parser.parse_language_file(file_name, lang):
            print(claim)
            all_claims.append(claim)

    print("-" * 50)
    print(f"SUCCESS: {len(all_claims)} claims parsed. Journals loaded: {len(parser.journals)}")
    print("-" * 50)
