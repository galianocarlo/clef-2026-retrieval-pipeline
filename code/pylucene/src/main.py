import os
import subprocess

def run_script(script_path):
    print(f"\n{'='*50}\nExecuting: {script_path}\n{'='*50}")
    # Runs the script as if executed from the terminal
    subprocess.run(["python", script_path], check=True)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. INDEX CHECK
    index_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "experiment", "index"))
    # If the directory doesn't exist or is empty, run the indexer
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print(">>> Index not found. Starting indexing...")
        run_script(os.path.join(BASE_DIR, "Index", "index.py"))
    else:
        print(">>> Index already exists")

    # 2. QRELS CHECK (Ground Truth)
    qrels_file = os.path.abspath(os.path.join(BASE_DIR, "..", "experiment", "docs", "qrels_dev.txt"))
    if not os.path.exists(qrels_file):
        print(">>> Qrels file not found. Generating it now...")
        run_script(os.path.join(BASE_DIR, "Evaluation", "generate_qrels.py"))
    else:
        print(">>> Qrels file already exists")

    # 3. SEARCH (We always run this, as it's the core of our experiments)
    print(">>> Starting the Searcher...")
    run_script(os.path.join(BASE_DIR, "Search", "search.py"))

    # 4. EVALUATION (We always run this to evaluate the latest search results)
    print(">>> Evaluating results...")
    run_script(os.path.join(BASE_DIR, "Evaluation", "evaluate.py"))

    print("\n PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()