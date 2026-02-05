import os
import json
import cv2

def is_inventory_useful(inventory):
    """
    Verifica se l'inventario contiene solo oggetti utili o è vuoto.
    """
    useful_items = {
        "crafting_table","oak_planks", "birch_planks", "spruce_planks", 
        "jungle_planks", "acacia_planks", "dark_oak_planks",
        "oak_log", "birch_log", "spruce_log", "jungle_log", 
        "acacia_log", "dark_oak_log"
    }
    # L'inventario è utile se è vuoto o contiene solo oggetti della lista
    return all(item["type"] in useful_items for item in inventory)

def get_cut_timestamp(jsonl_path):
    """
    Legge un file JSONL e restituisce il tick e il tempo relativo in cui tagliare il video.
    """
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        print(f"Errore di decodifica UTF-8 alla posizione {e.start}. Prova a verificare l'encoding del file.")
        return None, None, False

    first_milli = None
    average_server_tick_duration = 0
    first_inventory_check = True
    is_useful = True

    for line in lines:
        data = json.loads(line)

        if first_milli is None:
            first_milli = data["milli"]

        server_tick_duration = data.get("serverTickDurationMs", 50.0)
        average_server_tick_duration = (average_server_tick_duration + server_tick_duration) / 2

        inventory = data.get("inventory", [])

        # Se il primo inventario è già inutile, scartiamo il video
        if first_inventory_check:
            is_useful = is_inventory_useful(inventory)
            first_inventory_check = False
            if not is_useful:
                print(f"Video inizia con un inventario non utile: {inventory}")
                return None, None, False

        if not is_inventory_useful(inventory):
            cut_tick = data["tick"]
            cut_time = (data["milli"] - first_milli) / 1000.0
            print(f"Inventario non utile trovato a tick {cut_tick}: {inventory}")
            print(f"Tempo relativo: {cut_time} secondi")
            return cut_tick, cut_time, True

    print("Tutti i tick contengono inventari utili.")
    return None, None, True

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cut_frame = int(cut_time * fps)
    print(f"FPS: {fps}, Frame totali: {total_frames}, Frame di taglio: {cut_frame}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret or current_frame >= cut_frame:
            break

        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Video tagliato salvato in: {output_path}")
    return cut_frame

def trim_jsonl(jsonl_path, output_jsonl_path, max_tick):
    """
    Taglia il file JSONL fino al tick specificato.
    """
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as infile:
            lines = infile.readlines()
    except UnicodeDecodeError as e:
        print(f"Errore di decodifica UTF-8 alla posizione {e.start}.")
        return

    trimmed_lines = []
    for line in lines:
        data = json.loads(line)
        if data["tick"] > max_tick:
            break
        trimmed_lines.append(line)

    with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(trimmed_lines)

    print(f"JSONL tagliato salvato in: {output_jsonl_path}")

def process_videos(input_folder, output_folder):
    """
    Processa i video e i relativi JSONL nella cartella di input.
    """
    os.makedirs(output_folder, exist_ok=True)

    video_paths = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    for video_name in video_paths:
        jsonl_name = video_name.replace(".mp4", ".jsonl")
        video_path = os.path.join(input_folder, video_name)
        jsonl_path = os.path.join(input_folder, jsonl_name)

        if not os.path.exists(jsonl_path):
            print(f"JSONL non trovato per {video_name}, salto...")
            continue

        last_tick, cut_time, is_useful = get_cut_timestamp(jsonl_path)

        if not is_useful:
            print(f"Scartato video {video_name} per contenere oggetti non utili sin dall'inizio.")
            continue

        if cut_time is None or last_tick is None:
            print(f"Nessun taglio necessario per {video_name}, salto...")
            continue

        output_video_path = os.path.join(output_folder, f"trimmed_{video_name}")
        cut_frame = trim_video(video_path, output_video_path, cut_time)

        output_jsonl_path = os.path.join(output_folder, f"trimmed_{jsonl_name}")
        trim_jsonl(jsonl_path, output_jsonl_path, last_tick)

if __name__ == "__main__":
    input_folder = "/home/dodod/FiaCraft/data/data-video/iron"       
    output_folder = "/home/dodod/FiaCraft/data/data-video/video_tagliati"
    process_videos(input_folder, output_folder)
