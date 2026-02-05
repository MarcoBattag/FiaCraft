import os
import json
import argparse

def generate_video_jsonl(input_dir, output_file="download_videos.json", base_url="https://openaipublic.blob.core.windows.net/minecraft-rl/"):
    """
    Genera un file JSON contenente gli URL dei video corrispondenti ai file JSONL.

    Args:
        input_dir (str): Directory contenente i file JSONL filtrati.
        output_file (str): Percorso del file JSON da generare.
        base_url (str): URL base per i video.
    """
    video_paths = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            # Ottieni il percorso relativo del video
            video_filename = filename.replace(".jsonl", ".mp4")
            video_paths.append(f"data/10.0/{video_filename}")

    # Crea il dizionario con il formato richiesto
    output_data = {
        "basedir": base_url,
        "relpaths": video_paths
    }

    # Salva il JSON nel file di output
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"File JSON generato con successo: {output_file}")
    print(f"Totale video da scaricare: {len(video_paths)}")

def main():
    parser = argparse.ArgumentParser(description="Genera un file JSON con gli URL dei video corrispondenti ai JSONL")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory contenente i file JSONL filtrati")
    args = parser.parse_args()

    generate_video_jsonl(args.input_dir)

if __name__ == "__main__":
    main()
