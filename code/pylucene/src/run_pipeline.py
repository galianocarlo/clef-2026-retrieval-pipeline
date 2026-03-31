import os
import subprocess

def run_script(script_path):
    print(f"\n{'='*50}\nEsecuzione di: {script_path}\n{'='*50}")
    # Esegue lo script come se lo lanciassi da terminale
    subprocess.run(["python", script_path], check=True)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. CONTROLLO INDICE
    index_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "experiment", "index"))
    # Se la cartella non esiste o è vuota, lancia l'indexer
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print(">>> L'indice non esiste. Avvio l'indicizzazione...")
        run_script(os.path.join(BASE_DIR, "Index", "index.py"))
    else:
        print(">>> L'indice esiste già. Salto l'indicizzazione per risparmiare tempo!")

    # 2. CONTROLLO QRELS (Soluzioni)
    qrels_file = os.path.abspath(os.path.join(BASE_DIR, "..", "experiment", "docs", "qrels_dev.txt"))
    if not os.path.exists(qrels_file):
        print(">>> File Qrels non trovato. Lo genero ora...")
        run_script(os.path.join(BASE_DIR, "Evaluation", "generate_qrels.py"))
    else:
        print(">>> File Qrels già presente. Salto la generazione.")

    # 3. RICERCA (Questa la lanciamo sempre, perché è il cuore dei nostri esperimenti)
    print(">>> Avvio la Ricerca (Searcher)...")
    run_script(os.path.join(BASE_DIR, "Search", "search.py"))

    # 4. VALUTAZIONE (La lanciamo sempre per vedere come è andata l'ultima ricerca)
    print(">>> Valutazione dei risultati...")
    run_script(os.path.join(BASE_DIR, "Evaluation", "evaluate.py"))

    print("\n✅ PIPELINE COMPLETATA CON SUCCESSO!")

if __name__ == "__main__":
    main()