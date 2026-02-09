import argparse
import random
import urllib.request
import os
import glob
import cv2
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm # Opzionale: per una barra di caricamento professionale

parser = argparse.ArgumentParser(description="Download OpenAI contractor datasets in Parallel")
parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")
parser.add_argument("--workers", type=int, default=8, help="Number of simultaneous downloads")

def relpaths_to_download(relpaths, output_dir):
    def read_json(file_name):
        with open(file_name.replace('mp4', 'jsonl'), 'r') as json_file:
            return json.loads('['+''.join(json_file.readlines()).replace('\n', ',')+']')

    data_path = '/'.join(relpaths[0].split('/')[:-1])
    non_defect = []
    
    print("Verifica file esistenti...")
    vid_files = glob.glob(os.path.join(output_dir, '*.mp4'))
    
    for vid_name in vid_files:
        try:
            # Verifica che il video sia leggibile e il json associato sia integro
            vid = cv2.VideoCapture(vid_name)
            read_json(vid_name)
            if vid.isOpened():
                non_defect.append(os.path.join(data_path, os.path.basename(vid_name)))
            vid.release()
        except:
            continue

    relpaths_set = set(relpaths)
    non_defect_set = set(non_defect)
    diff_to_download = list(relpaths_set.difference(non_defect_set))
    
    print(f'Totale index: {len(relpaths)} | Già presenti: {len(non_defect)} | Da scaricare: {len(diff_to_download)}')
    return diff_to_download

def download_file_pair(task_info):
    """Funzione eseguita dai singoli thread"""
    url, outpath, jsonl_url, jsonl_outpath = task_info
    filename = os.path.basename(outpath)
    
    try:
        # Download MP4
        urllib.request.urlretrieve(url, outpath)
        
        # Download JSONL
        try:
            urllib.request.urlretrieve(jsonl_url, jsonl_outpath)
            return f"OK: {filename}"
        except Exception as e:
            if os.path.exists(outpath):
                os.remove(outpath)
            return f"ERRORE JSONL per {filename}: {e}"
            
    except Exception as e:
        return f"ERRORE MP4 per {filename}: {e}"

def main(args):
    # Caricamento dati dal file JSON index
    with open(args.json_file, "r") as f:
        content = f.read()
        # eval è usato perché l'indice originale OpenAI è spesso formattato come dizionario Python
        data = eval(content)
    
    basedir = data["basedir"]
    relpaths = data["relpaths"]

    random.shuffle(relpaths)

    if args.num_demos is not None:
        relpaths = relpaths[:args.num_demos]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Filtra i file già scaricati
    to_download = relpaths_to_download(relpaths, args.output_dir)

    # Preparazione della lista dei task per il multithreading
    tasks = []
    for relpath in to_download:
        url = basedir + relpath
        filename = os.path.basename(relpath)
        outpath = os.path.join(args.output_dir, filename)
        
        jsonl_url = url.replace(".mp4", ".jsonl")
        jsonl_outpath = outpath.replace(".mp4", ".jsonl")
        
        tasks.append((url, outpath, jsonl_url, jsonl_outpath))

    print(f"Avvio download parallelo con {args.workers} workers...")

    # Esecuzione parallela
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Uso list() per forzare l'esecuzione del generatore e monitorare i progressi
        results = list(tqdm(executor.map(download_file_pair, tasks), total=len(tasks), desc="Download in corso"))

    # Breve riassunto finale
    errors = [r for r in results if "ERRORE" in r]
    if errors:
        print(f"\nCompletato con {len(errors)} errori. Esempio:")
        for e in errors[:5]:
            print(f"  - {e}")
    else:
        print("\nTutti i download completati con successo!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)