import os
import json
import cv2

def has_wooden_pickaxe(inventory):
    """
    Verifica se l'inventario contiene un piccone di legno.
    """
    target_item = "wooden_pickaxe"
    # Cerca l'oggetto nell'inventario
    return any(item.get("type") == target_item for item in inventory)

def get_cut_timestamp(jsonl_path):
    """
    Legge un file JSONL e restituisce il tick e il tempo in cui il piccone viene creato.
    Restituisce is_valid=False se il piccone non viene mai creato o se è già presente all'inizio.
    """
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        print(f"Errore di decodifica UTF-8 alla posizione {e.start}.")
        return None, None, False

    first_milli = None
    first_inventory_check = True

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if first_milli is None:
            first_milli = data["milli"]

        inventory = data.get("inventory", [])

        # Controllo iniziale: se abbiamo già il piccone al primo frame, il video è inutile
        # (vogliamo imparare a costruirlo, non averlo già)
        if first_inventory_check:
            if has_wooden_pickaxe(inventory):
                print(f"Video inizia già con un piccone di legno. Scartato.")
                return None, None, False
            first_inventory_check = False

        # Verifica se il piccone è apparso
        if has_wooden_pickaxe(inventory):
            cut_tick = data["tick"]
            cut_time = (data["milli"] - first_milli) / 1000.0
            print(f"Piccone di legno costruito al tick {cut_tick}!")
            print(f"Tempo relativo: {cut_time:.2f} secondi")
            return cut_tick, cut_time, True

    print("Il piccone di legno non è mai stato costruito in questo video.")
    return None, None, False

def trim_video(video_path, output_path, cut_time):
    """
    Taglia il video fino al punto specificato.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calcolo frame di taglio
    cut_frame = int(cut_time * fps)
    
    # Aggiungiamo un piccolo buffer (es. 1 secondo o 20 frame) per vedere l'oggetto nell'inventario
    # Rimuovi la riga sotto se vuoi tagliare all'istante esatto
    cut_frame += int(fps * 1.0) 

    print(f"Taglio al frame: {cut_frame}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret or current_frame > cut_frame: # Modificato in > per includere il frame finale
            break

        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Video salvato in: {output_path}")
    return cut_frame

def trim_jsonl(jsonl_path, output_jsonl_path, max_tick):
    """
    Taglia il file JSONL fino al tick specificato.
    """
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as infile:
            lines = infile.readlines()
    except UnicodeDecodeError as e:
        print(f"Errore: {e}")
        return

    trimmed_lines = []
    # Buffer di tick extra per corrispondere al buffer video (opzionale, qui taglio esatto al tick rilevato)
    # Se vuoi includere qualche tick dopo, aumenta max_tick
    
    for line in lines:
        try:
            data = json.loads(line)
            if data["tick"] > max_tick + 20: # +20 tick di margine (circa 1 sec)
                break
            trimmed_lines.append(line)
        except json.JSONDecodeError:
            continue

    with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(trimmed_lines)

    print(f"JSONL salvato in: {output_jsonl_path}")

def process_videos(input_folder, output_folder):
    """
    Processa i video e i relativi JSONL nella cartella di input.
    """
    os.makedirs(output_folder, exist_ok=True)

    video_paths = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    
    count_processed = 0
    
    for video_name in video_paths:
        jsonl_name = video_name.replace(".mp4", ".jsonl")
        video_path = os.path.join(input_folder, video_name)
        jsonl_path = os.path.join(input_folder, jsonl_name)

        if not os.path.exists(jsonl_path):
            print(f"JSONL non trovato per {video_name}, salto...")
            continue

        print(f"\nProcessing: {video_name}...")
        
        # Ottieni punto di taglio (o False se il video non è valido)
        last_tick, cut_time, is_valid = get_cut_timestamp(jsonl_path)

        if not is_valid:
            print(f"Video {video_name} SCARTATO (Obiettivo non raggiunto o già presente).")
            continue

        output_video_path = os.path.join(output_folder, f"trimmed_{video_name}")
        trim_video(video_path, output_video_path, cut_time)

        output_jsonl_path = os.path.join(output_folder, f"trimmed_{jsonl_name}")
        trim_jsonl(jsonl_path, output_jsonl_path, last_tick)
        
        count_processed += 1

    print(f"\n--- Finito ---")
    print(f"Video processati e salvati correttamente: {count_processed}/{len(video_paths)}")

if __name__ == "__main__":
    # Assicurati di aggiornare i percorsi se necessario
    input_folder = "*/FiaCraft/data/data-video/iron"       
    output_folder = "*/FiaCraft/data/data-video/video_tagliati_pickaxe"
    process_videos(input_folder, output_folder)
