# data_loader.py - VERSIONE SEQUENTIAL (Fixed per VPT)
import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event
import numpy as np
import cv2
from openai_vpt.agent import resize_image, AGENT_RESOLUTION

# Configurazione
QUEUE_TIMEOUT = 60
CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC", "key.keyboard.s" :"back", "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward", "key.keyboard.1" :"hotbar.1", "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3", "key.keyboard.4" :"hotbar.4", "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6", "key.keyboard.7" :"hotbar.7", "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9", "key.keyboard.e" :"inventory", "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left", "key.keyboard.d" :"right", "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint", "key.keyboard.f" :"swapHands",
}

NOOP_ACTION = {
    "ESC": 0, "back": 0, "drop": 0, "forward": 0, "hotbar.1": 0, "hotbar.2": 0,
    "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0,
    "hotbar.8": 0, "hotbar.9": 0, "inventory": 0, "jump": 0, "left": 0, "right": 0,
    "sneak": 0, "sprint": 0, "swapHands": 0, "camera": np.array([0, 0]),
    "attack": 0, "use": 0, "pickItem": 0,
}

MINEREC_ORIGINAL_HEIGHT_PX = 720
CAMERA_SCALER = 360.0 / 2400.0

def json_action_to_env_action(json_action):
    env_action = NOOP_ACTION.copy()
    env_action["camera"] = np.array([0, 0])
    is_null_action = True
    
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180: camera_action[0] = 0
        if abs(camera_action[1]) > 180: camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons: env_action["attack"] = 1; is_null_action = False
    if 1 in mouse_buttons: env_action["use"] = 1; is_null_action = False
    if 2 in mouse_buttons: env_action["pickItem"] = 1; is_null_action = False

    return env_action, is_null_action

def composite_images_with_alpha(image1, image2, alpha, x, y):
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0: return
    try:
        alpha_slice = alpha[:ch, :cw]
        image1[y:y + ch, x:x + cw, :] = (
            image1[y:y + ch, x:x + cw, :] * (1 - alpha_slice) + image2[:ch, :cw, :] * alpha_slice
        ).astype(np.uint8)
    except: pass

def data_loader_worker(tasks_queue, output_queue, quit_workers_event, seq_len):
    """
    Worker modificato per restituire SEQUENZE intere, non frame singoli.
    """
    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    if cursor_image is not None:
        cursor_image = cursor_image[:16, :16, :]
        cursor_alpha = cursor_image[:, :, 3:] / 255.0
        cursor_image = cursor_image[:, :, :3]
    else:
        # Fallback se manca il cursore
        cursor_alpha = None

    while True:
        task = tasks_queue.get()
        if task is None: break
        
        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)
        
        # Buffer per la sequenza
        frames_buffer = []
        actions_buffer = []
        first_flags_buffer = [] # True se è l'inizio del video o dopo un taglio
        
        attack_is_stuck = False
        last_hotbar = 0

        try:
            with open(json_path, encoding='utf-8', errors='ignore') as json_file:
                json_lines = json_file.readlines()
                json_data = "[" + ",".join(json_lines) + "]"
                json_data = json.loads(json_data)
        except Exception as e:
            print(f"Skipping broken JSON {json_path}: {e}")
            output_queue.put(None) # Segnala fine task
            continue

        is_first_frame = True

        for i in range(len(json_data)):
            if quit_workers_event.is_set(): break
            step_data = json_data[i]

            # Gestione click mouse "incastrati"
            if i == 0:
                if step_data["mouse"]["newButtons"] == [0]: attack_is_stuck = True
            elif attack_is_stuck:
                if 0 in step_data["mouse"]["newButtons"]: attack_is_stuck = False
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [b for b in step_data["mouse"]["buttons"] if b != 0]

            action, is_null_action = json_action_to_env_action(step_data)
            
            # Hotbar logic
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            ret, frame = video.read()
            if ret:
                if is_null_action: continue # Salta frame nulli ma mantiene continuità video (rischio desync, ma standard in minerl)
                
                # Render cursore
                if step_data["isGuiOpen"] and cursor_image is not None:
                    camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                    cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                    composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
                
                # Preprocessing frame
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)
                
                # Accumula nel buffer
                frames_buffer.append(frame)
                actions_buffer.append(action)
                first_flags_buffer.append(is_first_frame)
                is_first_frame = False

                # Se abbiamo raggiunto la lunghezza della sequenza, invia il pacchetto
                if len(frames_buffer) == seq_len:
                    output_queue.put((trajectory_id, np.array(frames_buffer), np.array(actions_buffer), np.array(first_flags_buffer)))
                    frames_buffer = []
                    actions_buffer = []
                    first_flags_buffer = []
            else:
                break
        
        video.release()
        output_queue.put(None) # Segnala fine video
        if quit_workers_event.is_set(): break
    
    output_queue.put(None) # Segnala fine worker

class DataLoader:
    def __init__(self, dataset_dir, n_workers=4, batch_size=4, n_epochs=1, seq_len=32, max_queue_size=8):
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seq_len = seq_len # Ora supportiamo seq_len!
        self.max_queue_size = max_queue_size
        
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids
        
        demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))

        self.demonstration_tuples = []
        for i in range(n_epochs):
            random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples

        self.task_queue = Queue()
        self.n_steps_processed = 0
        
        # Riempi coda task
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                    seq_len # Passiamo seq_len al worker
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        for process in self.processes:
            process.start()
        
        # Buffer interno per gestire i None (fine video) restituiti dai worker
        self.worker_finished_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_actions = []
        batch_episode_id = []
        batch_first = []

        collected = 0
        while collected < self.batch_size:
            # Round-robin sui worker
            worker_idx = self.n_steps_processed % self.n_workers
            
            try:
                workitem = self.output_queues[worker_idx].get(timeout=QUEUE_TIMEOUT)
            except:
                # Se timeout, prova il prossimo worker
                self.n_steps_processed += 1
                continue

            if workitem is None:
                # Un video o un worker è finito
                # Nota: Una gestione robusta richiederebbe di contare quanti worker sono morti definitivamente
                # Per ora saltiamo al prossimo
                self.n_steps_processed += 1
                continue

            # Unpack dei dati (ora sono SEQUENZE, non singoli frame)
            # workitem è (trajectory_id, frames_seq, actions_seq, first_flags_seq)
            trajectory_id, frames, actions, firsts = workitem
            
            # Qui frames ha shape (seq_len, H, W, C)
            batch_frames.append(frames)
            batch_actions.append(actions)
            batch_episode_id.append(trajectory_id)
            batch_first.append(firsts)
            
            collected += 1
            self.n_steps_processed += 1

        return batch_frames, batch_actions, batch_episode_id, batch_first

    def __del__(self):
        if hasattr(self, 'quit_workers_event'):
            self.quit_workers_event.set()
        if hasattr(self, 'processes'):
            for process in self.processes:
                process.terminate()
                process.join()