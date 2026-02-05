import os
import json
import argparse

def is_inventory_empty(jsonl_path):
    """
    Controlla se il primo frame del file JSONL ha un inventario vuoto.

    Args:
        jsonl_path (str): Percorso del file JSONL.

    Returns:
        bool: True se l'inventario è vuoto, False altrimenti.
    """
    try:
        with open(jsonl_path, 'r') as file:
            first_line = file.readline()
            if not first_line:
                return False  # File vuoto, da considerare come "non valido"
            data = json.loads(first_line)
            inventory = data.get("inventory", [])
            return len(inventory) == 0  # Controlla se la lista è vuota
    except Exception as e:
        print(f"Errore nel leggere il file {jsonl_path}: {e}")
        return False

def filter_jsonl_files(input_dir, output_dir):
    """
    Filtra i file JSONL che iniziano con un inventario non vuoto.

    Args:
        input_dir (str): Directory contenente i file JSONL.
        output_dir (str): Directory in cui salvare i file validi.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_dir, filename)
            if is_inventory_empty(file_path):
                print(f"File valido: {filename}")
                os.rename(file_path, os.path.join(output_dir, filename))
            else:
                print(f"File scartato: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Filtra i file JSONL con inventario iniziale vuoto")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory contenente i file JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory in cui salvare i file filtrati")
    args = parser.parse_args()

    filter_jsonl_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
