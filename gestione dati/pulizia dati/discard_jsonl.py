import os
import json
import argparse
import shutil


def is_inventory_empty(jsonl_path):
    """
    Controlla se il primo frame del file JSONL ha un inventario vuoto.
    """
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as file:
            first_line = file.readline()

            if not first_line:
                return False

            data = json.loads(first_line)
            inventory = data.get("inventory", [])

            return len(inventory) == 0

    except Exception as e:
        print(f"Errore nel leggere {jsonl_path}: {e}")
        return False


def filter_dataset(input_dir, output_dir):
    """
    Copia nella cartella di output solo le coppie video+jsonl
    che iniziano con inventario vuoto.
    """
    os.makedirs(output_dir, exist_ok=True)

    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    kept = 0
    discarded = 0

    for jsonl_name in jsonl_files:
        jsonl_path = os.path.join(input_dir, jsonl_name)

        video_name = jsonl_name.replace(".jsonl", ".mp4")
        video_path = os.path.join(input_dir, video_name)

        if not os.path.exists(video_path):
            print(f"Video mancante per {jsonl_name}, salto...")
            continue

        if is_inventory_empty(jsonl_path):
            print(f"KEEP: {jsonl_name}")

            shutil.copy(jsonl_path, os.path.join(output_dir, jsonl_name))
            shutil.copy(video_path, os.path.join(output_dir, video_name))

            kept += 1
        else:
            print(f"DISCARD: {jsonl_name}")
            discarded += 1

    print("\n--- RISULTATO ---")
    print(f"Tenuti: {kept}")
    print(f"Scartati: {discarded}")
    print(f"Totale: {kept + discarded}")


def main():
    parser = argparse.ArgumentParser(
        description="Filtra il dataset MineRL mantenendo solo i video con inventario iniziale vuoto"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    filter_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
