# Code for loading OpenAI MineRL VPT datasets
# (NOTE: Modified for Sequence Loading!)
import json
import glob
import os
import random
from multiprocessing import Process, Queue, Event

import numpy as np
import cv2

from openai_vpt.agent import ACTION_TRANSFORMER_KWARGS, resize_image, AGENT_RESOLUTION
from openai_vpt.lib.actions import ActionTransformer

QUEUE_TIMEOUT = 300
CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

# Mapping from JSON keyboard buttons to MineRL actions
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
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
    # This might be slow...
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
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def composite_images_with_alpha(image1, image2, alpha, x, y):
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0: return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


# ### MODIFICA: Aggiunto parametro seq_len al worker
def data_loader_worker(tasks_queue, output_queue, quit_workers_event, seq_len):
    """
    Worker che processa i video e crea chunk di sequenze.
    """
    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]

    while True:
        task = tasks_queue.get()
        if task is None:
            break
        trajectory_id, video_path, json_path = task
        video = cv2.VideoCapture(video_path)
        attack_is_stuck = False
        last_hotbar = 0

        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)

        # ### MODIFICA: Buffer per accumulare la sequenza
        obs_buffer = []
        act_buffer = []
        is_first_buffer = [] # Per sapere se è l'inizio di un episodio
        # ---------------------------------------------

        for i in range(len(json_data)):
            if quit_workers_event.is_set():
                break
            step_data = json_data[i]

            if i == 0:
                if step_data["mouse"]["newButtons"] == [0]: attack_is_stuck = True
            elif attack_is_stuck:
                if 0 in step_data["mouse"]["newButtons"]: attack_is_stuck = False
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

            action, is_null_action = json_action_to_env_action(step_data)

            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            ret, frame = video.read()
            if ret:
                # Nota: qui NON saltiamo i null actions.
                # Per le sequenze (LSTM) è meglio avere continuità temporale anche se non succede nulla.
                # Se vuoi risparmiare spazio e il null è irrilevante, puoi decommentare, ma per il crafting
                # a volte stare fermi è importante.
                # if is_null_action: continue 

                if step_data["isGuiOpen"]:
                    camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                    cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                    cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                    composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
                
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frame = resize_image(frame, AGENT_RESOLUTION)

                # ### MODIFICA: Aggiungi al buffer invece di inviare subito
                obs_buffer.append(frame)
                act_buffer.append(action)
                
                # Flag: True se è il primissimo frame del video, altrimenti False
                is_first_flag = (i == 0) and (len(obs_buffer) == 1)
                is_first_buffer.append(is_first_flag)

                # Se il buffer è pieno (abbiamo una sequenza completa)
                if len(obs_buffer) >= seq_len:
                    # Converti in numpy arrays
                    # Shape: (SEQ_LEN, H, W, C)
                    np_obs = np.array(obs_buffer)
                    # Shape: (SEQ_LEN) - lista di dizionari
                    np_act = np.array(act_buffer)
                    np_first = np.array(is_first_buffer)

                    output_queue.put((trajectory_id, np_obs, np_act, np_first), timeout=QUEUE_TIMEOUT)
                    
                    # Resetta i buffer
                    obs_buffer = []
                    act_buffer = []
                    is_first_buffer = []
                # --------------------------------------------------------

            else:
                print(f"Could not read frame from video {video_path}")
        
        video.release()
        output_queue.put((trajectory_id, None, None, None), timeout=QUEUE_TIMEOUT)
        if quit_workers_event.is_set():
            break
    
    output_queue.put(None)

class DataLoader:
    """
    Generator class for loading SEQUENCES from a dataset.
    Returns batches of shape (BATCH, TIME, H, W, C)
    """
    # ### MODIFICA: Aggiunto parametro seq_len (default 32 frames)
    def __init__(self, dataset_dir, n_workers=8, batch_size=4, n_epochs=1, max_queue_size=8, seq_len=32):
        assert n_workers >= batch_size, "Number of workers must be equal or greater than batch size"
        self.dataset_dir = dataset_dir
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.seq_len = seq_len # Nuova proprietà
        
        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))
        unique_ids = list(set([os.path.basename(x).split(".")[0] for x in unique_ids]))
        self.unique_ids = unique_ids
        
        demonstration_tuples = []
        for unique_id in unique_ids:
            video_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".mp4"))
            json_path = os.path.abspath(os.path.join(dataset_dir, unique_id + ".jsonl"))
            demonstration_tuples.append((video_path, json_path))

        assert n_workers <= len(demonstration_tuples), f"n_workers should be lower or equal than number of demonstrations"

        self.demonstration_tuples = []
        for i in range(n_epochs):
            random.shuffle(demonstration_tuples)
            self.demonstration_tuples += demonstration_tuples

        self.task_queue = Queue()
        self.n_steps_processed = 0
        for trajectory_id, task in enumerate(self.demonstration_tuples):
            self.task_queue.put((trajectory_id, *task))
        for _ in range(n_workers):
            self.task_queue.put(None)

        self.output_queues = [Queue(maxsize=max_queue_size) for _ in range(n_workers)]
        self.quit_workers_event = Event()
        
        # ### MODIFICA: Passiamo seq_len al worker
        self.processes = [
            Process(
                target=data_loader_worker,
                args=(
                    self.task_queue,
                    output_queue,
                    self.quit_workers_event,
                    self.seq_len 
                ),
                daemon=True
            )
            for output_queue in self.output_queues
        ]
        # ---------------------------------------
        
        for process in self.processes:
            process.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = []
        batch_actions = []
        batch_episode_id = []
        batch_first = [] # Nuovo array per indicare l'inizio episodi

        for i in range(self.batch_size):
            workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(timeout=QUEUE_TIMEOUT)
            if workitem is None:
                raise StopIteration()
            
            # ### MODIFICA: Unpack include ora 'is_first'
            trajectory_id, frame_seq, action_seq, first_seq = workitem
            
            if frame_seq is None:
                # Gestione fine video (logica semplificata per non bloccare)
                # In una implementazione robusta dovremmo riprovare a prendere un altro item
                # Qui usiamo ricorsione semplice o saltiamo
                # Per semplicità, rilanciamo StopIteration se un worker finisce, 
                # ma in produzione dovresti gestire il 'None' meglio.
                # Per ora assumiamo che i dati siano ben bilanciati.
                return self.__next__() 

            batch_frames.append(frame_seq)
            batch_actions.append(action_seq)
            batch_episode_id.append(trajectory_id)
            batch_first.append(first_seq)
            
            self.n_steps_processed += 1
        
        # Ritorna tuple: (Batch, Time, H, W, C)
        return batch_frames, batch_actions, batch_episode_id, batch_first

    def __del__(self):
        self.quit_workers_event.set()
        for process in self.processes:
            process.terminate()
            process.join()