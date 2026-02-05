import cv2
import os
import glob
import json
import argparse

# Funzione per interpretare i valori booleani
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Valore booleano invalido: {value}")


def mirror_video(input_path, output_path, json_data):
    # Apri il video originale
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {input_path}")
        return

    # Ottieni le proprietà del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per output in formato .mp4

    if fps == 0 or width == 0 or height == 0:
        print(f"Errore: il video {input_path} ha FPS o dimensioni non validi.")
        cap.release()
        return

    # Configura il writer per salvare il video specchiato
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Controlla isGuiOpen per decidere se specchiare il frame
        if frame_index < len(json_data) and not json_data[frame_index].get("isGuiOpen", False):
            mirrored_frame = cv2.flip(frame, 1)  # Specchia solo se isGuiOpen è False
        else:
            mirrored_frame = frame  # Non specchiare se isGuiOpen è True

        out.write(mirrored_frame)
        frame_index += 1

    # Chiudi tutto
    cap.release()
    out.release()
    print(f"Video specchiato salvato in: {output_path}")


def mirror_json(input_json_path, output_json_path, video_path):
    cap = cv2.VideoCapture(video_path)
    width = 1280  # Dimensione dello schermo
    cap.release()

    with open(input_json_path, "r") as infile:
        lines = infile.readlines()

    mirrored_lines = []
    for line in lines:
        data = json.loads(line)

        # Specchia solo se isGuiOpen è False
        if not data.get("isGuiOpen", False):
            if "mouse" in data:
                data["mouse"]["dx"] = -data["mouse"]["dx"]
                data["mouse"]["scaledX"] = -data["mouse"]["scaledX"]
                data["mouse"]["x"] = width - data["mouse"]["x"]

            if "keyboard" in data and "keys" in data["keyboard"]:
                keys = data["keyboard"]["keys"]
                new_keys = []
                new_chars = []
                for key in keys:
                    if key == "key.keyboard.a":
                        new_keys.append("key.keyboard.d")
                        new_chars.append("d")
                    elif key == "key.keyboard.d":
                        new_keys.append("key.keyboard.a")
                        new_chars.append("a")
                    else:
                        new_keys.append(key)
                        if "chars" in data["keyboard"]:
                            new_chars.append(data["keyboard"]["chars"])

                data["keyboard"]["keys"] = new_keys
                if "chars" in data["keyboard"]:
                    data["keyboard"]["chars"] = "".join(new_chars)

            if "hotbar" in data:
                data["hotbar"] = 8 - data["hotbar"]

        mirrored_lines.append(json.dumps(data))

    with open(output_json_path, "w") as outfile:
        outfile.write("\n".join(mirrored_lines) + "\n")
    print(f"JSONL specchiato salvato in: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Specchia video e JSONL.")
    parser.add_argument("--input_folder", type=str, required=True, help="Cartella di input contenente video e JSONL.")
    parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output per i video e JSONL specchiati.")
    parser.add_argument("--overwrite", type=str_to_bool, default=True, help="Sovrascrive i file specchiati se esistono già (default: True).")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    overwrite = args.overwrite

    os.makedirs(output_folder, exist_ok=True)

    video_paths = glob.glob(os.path.join(input_folder, "*.mp4"))

    for video_path in video_paths:
        video_name = os.path.basename(video_path)

        if video_name.startswith("mirrored_"):
            continue

        mirrored_video_path = os.path.join(output_folder, f"mirrored_{video_name}")

        if not overwrite and os.path.exists(mirrored_video_path):
            continue

        json_name = video_name.replace(".mp4", ".jsonl")
        input_json_path = os.path.join(input_folder, json_name)
        mirrored_json_path = os.path.join(output_folder, f"mirrored_{json_name}")

        if os.path.exists(input_json_path):
            with open(input_json_path, "r") as f:
                json_data = [json.loads(line) for line in f]
            mirror_video(video_path, mirrored_video_path, json_data)
            mirror_json(input_json_path, mirrored_json_path, video_path)
        else:
            print(f"Attenzione: JSONL non trovato per {video_name}")


if __name__ == "__main__":
    main()
