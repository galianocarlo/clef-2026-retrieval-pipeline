import os
import sys
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    clear_screen()
    print("="*70)
    print(" MULTILINGUAL CLINICAL RETRIEVAL PIPELINE - INTERACTIVE RUNNER")
    print("="*70)
    print(" 1 - V1: Lexical Baseline (BM25 Only)")
    print(" 2 - V2: Statistical PRF")
    print(" 3 - V3: Hybrid + Chunking (e5-base & mMiniLMv2)")
    print(" 4 - V4: SOTA Hybrid Architecture (BGE-M3)")
    print("="*70)

def run_command(command, env=None):
    print(f"\n[Executing] {' '.join(command)} ...")
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError:
        print("\n[ERROR] Script execution failed. Check the logs above.")
        sys.exit(1)

def main():
    print_menu()
    choice = input("\nSelect a version (1-4) or 'q' to quit: ").strip().lower()
    if choice == 'q': sys.exit(0)
    if choice not in ['1', '2', '3', '4']: sys.exit(1)

    print("\n[A] Do you want to index the documents? (Press 'y' if you changed FAISS version)")
    do_index = input("(y/n): ").strip().lower() == 'y'

    print("\n[B] Which split do you want to run the search on?")
    print(" - 'train': To test on the training set (evaluate metrics available)")
    print(" - 'dev'  : To test on the validation set (evaluate metrics available)")
    print(" - 'test' : To generate the final run for submission (NO local evaluation)")
    split_choice = input("(train/dev/test): ").strip().lower()
    
    if split_choice not in ['train', 'dev', 'test']:
        print("Invalid split choice.")
        sys.exit(1)

    current_env = os.environ.copy()
    
    # Automatic Mapping of Scripts and Output files
    if choice == '1':
        index_script = ["python", "src/Index/index.py"]
        search_script = ["python", "src/Search/search_v1.py", "--split", split_choice]
        run_filename = f"run-v1_{split_choice}.txt"
    elif choice == '2':
        index_script = ["python", "src/Index/index.py"]
        search_script = ["python", "src/Search/search_v2.py", "--split", split_choice]
        run_filename = f"run-v2_{split_choice}.txt"
    elif choice == '3':
        index_script = ["python", "src/Index/index_faiss.py"]
        search_script = ["python", "src/Search/search_v3.py", "--split", split_choice]
        current_env["FAISS_MODEL"] = "intfloat/multilingual-e5-base"
        run_filename = f"run-V3_{split_choice}.txt"
    elif choice == '4':
        index_script = ["python", "src/Index/index_faiss.py"]
        search_script = ["python", "src/Search/search_v4.py", "--split", split_choice]
        current_env["FAISS_MODEL"] = "BAAI/bge-m3"
        run_filename = f"run-V4_{split_choice}.txt"

    # PHASE 1: INDEX
    if do_index:
        if choice in ['3', '4']: 
            print("\n Removing old corpus.faiss file to prevent dimension conflicts...")
            os.system("rm -f experiment/index/corpus.faiss")
        print("\n--- PHASE 1: INDEXING ---")
        run_command(index_script, env=current_env)
    
    # PHASE 2: SEARCH
    print(f"\n--- PHASE 2: SEARCHING ON {split_choice.upper()} ---")
    run_command(search_script, env=current_env)
    
    # PHASE 3: EVALUATE (Only on TRAIN and DEV)
    if split_choice in ["train", "dev"]:
        print(f"\n[C] Do you want to evaluate the generated results on the {split_choice.upper()} set?")
        if input("(y/n): ").strip().lower() == 'y':
            print("\n--- PHASE 3: METRICS EVALUATION ---")
            run_path = f"experiment/runs/{run_filename}"
            
            evaluate_script = ["python", "src/Evaluation/evaluate.py", run_path, "--split", split_choice]
            run_command(evaluate_script, env=current_env)
    else:
        print("\n TEST run generated successfully. No local evaluation possible.")

if __name__ == "__main__":
    main()