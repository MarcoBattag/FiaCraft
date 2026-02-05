import minerl
import gym
import cv2
import json
import pygame
import numpy as np
import os
from utils.generate_file import generate_file_name
import sys
import os
sys.path.append(os.path.abspath('../../'))

import register_envs  # Importa il file di registrazione degli ambienti

img_scaling = 0.25

samples = 1
output_video_path = os.getcwd()

for i in range(samples):
    # Initialize pygame
    pygame.init()

    frames = []

    file_base_name = f"FIA_{generate_file_name()}"
    
    # Constants
    OUTPUT_VIDEO_FILE = f"./in_costruzione/{file_base_name}.mp4"
    ACTION_LOG_FILE = f"./in_costruzione/{file_base_name}.jsonl"
    FPS = 20
    RESOLUTION = (1280, 720)  # Resolution at which to capture and save the video
    VIDEO_OUT_RESOLUTION = (640, 360)  # Resolution at which to capture and save the video
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption('Minecraft')
    SENS = 0.2
    SENS_INVERSA = 2.5

    # Set up the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS, VIDEO_OUT_RESOLUTION)

    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)  # Center the mouse
    pygame.event.set_grab(True)

    prev_mouse_x, prev_mouse_y = pygame.mouse.get_pos()
    reg_mouse_x, reg_mouse_y = (screen.get_width() // 2, screen.get_height() // 2)
    reg_mouse_x_gui_open, reg_mouse_y_gui_open = (screen.get_width() // 2, screen.get_height() // 2)

    # Mapping from pygame key to action
    key_to_action_mapping = {
        pygame.K_w: {'forward': 1},
        pygame.K_s: {'back': 1},
        pygame.K_a: {'left': 1},
        pygame.K_d: {'right': 1},
        pygame.K_SPACE: {'jump': 1},
        pygame.K_1: {'hotbar.1': 1},
        pygame.K_2: {'hotbar.2': 1},
        pygame.K_3: {'hotbar.3': 1},
        pygame.K_4: {'hotbar.4': 1},
        pygame.K_5: {'hotbar.5': 1},
        pygame.K_6: {'hotbar.6': 1},
        pygame.K_7: {'hotbar.7': 1},
        pygame.K_8: {'hotbar.8': 1},
        pygame.K_9: {'hotbar.9': 1},
        pygame.K_LSHIFT: {'sprint': 1},
        pygame.K_LCTRL: {'sneak': 1},
        pygame.K_g: {'drop': 1},
        pygame.K_e: {'inventory': 1},
        pygame.K_f: {'swapHands': 1},
        pygame.K_t: {'pickItem': 1}
    }
    # ... movement keys, jump, crouch, sprint, hotbar, attack, use, inventory, drop, swaphands, pickitem
    # Mapping from mouse button to action
    mouse_to_action_mapping = {
        0: {'attack': 1},      # Left mouse button
        2: {'use': 1}    # Right mouse button
        # Add more if needed
    }
    
    action_log = []

    # Initialize the Minecraft environment
    env = gym.make('FIA-Treechop-v0')

    #env.seed(2143)
  
    obs = env.reset()

    done = False
    tick = 0
    server_tick = 0
    server_tick_interval = 50
    last_server_tick_time = pygame.time.get_ticks()
    """
    # action_space = {"ESC": 0,
    #          "noop": [], 
    #          "attack": 0, 
    #          "back": 0, 
    #          "drop": 0, 
    #          "forward": 0, 
    #          "hotbar.1": 0, 
    #          "hotbar.2": 0, 
    #          "hotbar.3": 0, 
    #          "hotbar.4": 0, 
    #          "hotbar.5": 0, 
    #          "hotbar.6": 0, 
    #          "hotbar.7": 0, 
    #          "hotbar.8": 0, 
    #          "hotbar.9": 0, 
    #          "inventory": 0, 
    #          "jump": 0, 
    #          "left": 0, 
    #          "right": 0, 
    #          "pickItem": 0, 
    #          "sneak": 0, 
    #          "sprint": 0, 
    #          "swapHands": 0, 
    #          "use": 0,
    #          "camera": [0.0, 0.0]}
    """
    
    isGuiOpen = False
    isGuiInventory = False
    isGuiTriggerActive = False
    current_hotbar = 0
    
    try:
        while not done:
            # Convert the observation to a format suitable for pygame display
            image = np.array(obs['pov'])
            # image = cvtColor(image, cv2)
            # Record the current frame
            
            # print(image.shape)
            # out_image = cv2.resize(image, (int(360 * img_scaling), int(640 * img_scaling)))
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imshow('temp', image)
            image = np.flip(image, axis=1)
            image = np.rot90(image)
            # image = image * 0.1 # <- brightness
            display_image = cv2.resize(image,(720,1280))
            display_image = pygame.surfarray.make_surface(display_image)
            screen.blit(display_image, (0, 0))
            pygame.display.update()
        
            keys = pygame.key.get_pressed()

            # Build the action dictionary
            action = {'noop': []}
            for key, act in key_to_action_mapping.items():
                if keys[key]:
                    action.update(act)
                    # Verifica se l'azione riguarda un cambio di hotbar
                    if key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, 
                               pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                        # Estrai il numero della hotbar dall'azione
                        current_hotbar = int(pygame.key.name(key)) - 1

            mouse_buttons = pygame.mouse.get_pressed()
            for idx, pressed in enumerate(mouse_buttons):
                if pressed:
                    action.update(mouse_to_action_mapping.get(idx, {}))
                    if idx == 2 and isGuiTriggerActive:  # Controlla se il tasto destro è premuto e il trigger è attivo
                        isGuiOpen = True

            # Get mouse movement
            mouse_x, mouse_y = pygame.mouse.get_pos()
            delta_x = mouse_x - prev_mouse_x 
            delta_y = mouse_y - prev_mouse_y
            action["camera"] = [(delta_y * SENS)* (-1), (delta_x * SENS)* (-1)]
            pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)
            prev_mouse_x, prev_mouse_y = screen.get_width() // 2, screen.get_height() // 2

            if isGuiOpen:
                # Aggiorna le coordinate GUI-specifiche
                reg_mouse_x_gui_open -= delta_x * SENS_INVERSA
                reg_mouse_y_gui_open -= delta_y * SENS_INVERSA

                # Limita le coordinate del cursore alla finestra
                reg_mouse_x_gui_open = max(0, min(reg_mouse_x_gui_open, RESOLUTION[0] - 1))
                reg_mouse_y_gui_open = max(0, min(reg_mouse_y_gui_open, RESOLUTION[1] - 1))
            else:
                # Aggiorna le coordinate normali
                reg_mouse_x -= delta_x * SENS_INVERSA
                reg_mouse_y -= delta_y * SENS_INVERSA  


            # Get keys pressed
            keys_pressed = [
                f"key.keyboard.{pygame.key.name(key)}"
                for key in key_to_action_mapping.keys() if keys[key]
            ]
            
            # Escludi il tasto `x` dalle chiavi registrate
            keys_pressed = [key for key in keys_pressed if key != "key.keyboard.x"]

            # Calcola `newKeys` confrontando con i tasti del ciclo precedente
            new_keys_pressed = [key for key in keys_pressed if key not in previous_keys]

            # Aggiorna i tasti precedenti, gestisce l'apertura e la chiusura della GUI e il tasto quit
            chars = ""
            previous_keys = keys_pressed.copy()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    char = event.unicode
                    if char:
                        chars += char  # Aggiungi il carattere digitato
                        key_name = f"key.keyboard.{char.lower()}"
                        if key_name not in new_keys_pressed and key_name != "key.keyboard.x":
                            new_keys_pressed.append(key_name)  # Aggiungi ai nuovi tasti
                    if event.key == pygame.K_e:
                        if(not isGuiTriggerActive):
                            isGuiOpen = not isGuiOpen
                            isGuiInventory = not isGuiInventory
                        else:
                            isGuiOpen = not isGuiOpen
                            isGuiTriggerActive = False
                            
                        reg_mouse_x, reg_mouse_y = (screen.get_width() // 2, screen.get_height() // 2)
                        reg_mouse_x_gui_open, reg_mouse_y_gui_open = (screen.get_width() // 2, screen.get_height() // 2)
                    elif event.key == pygame.K_x:  # Attiva/disattiva il trigger
                        isGuiTriggerActive = not isGuiTriggerActive
                        print(f"Trigger GUI: {'attivo' if isGuiTriggerActive else 'disattivo'}")

            # Calcola il server tick
            current_time = pygame.time.get_ticks()
            elapsed_time = current_time - last_server_tick_time  # Tempo trascorso in ms
            server_tick = elapsed_time // server_tick_interval  # Ogni 50 ms corrisponde a un tick del server
            
            # Converte l'inventario nel formato richiesto
            inventory_data = obs.get('inventory', {})
            formatted_inventory = []

            # Itera sull'inventario e formatta gli oggetti con quantità >= 1
            for item_name, quantity in inventory_data.items():
                if int(quantity) >= 1:  # Controlla se la quantità è >= 1
                    formatted_inventory.append({
                        "type": str(item_name),  # Nome oggetto
                        "quantity": int(quantity)  # Quantità
                    })
                    
            # Variabili per tracciare i tasti del mouse
            current_buttons = []  # Pulsanti attualmente premuti
            new_buttons = []  # Pulsanti appena premuti

            # Mappa personalizzata per i pulsanti del mouse
            mouse_button_mapping = {
                0: 0,  # Tasto sinistro
                2: 1   # Tasto destro
            }

            # Conversione in formato richiesto
            for idx, pressed in enumerate(mouse_buttons):
                if pressed and idx in mouse_button_mapping:
                    current_buttons.append(mouse_button_mapping[idx])  # Aggiungi il pulsante mappato

            # Calcola `newButtons` confrontando con i pulsanti del frame precedente
            if 'previous_buttons' in locals():
                new_buttons = [btn for btn in current_buttons if btn not in previous_buttons]

            # Aggiorna i pulsanti del frame precedente
            previous_buttons = current_buttons.copy()
            print(f"{pygame.mouse.get_pos()} posizione registrata (x,y): {reg_mouse_x_gui_open} {reg_mouse_y_gui_open}")

            registerAction = {
                "mouse": {
                    "x": reg_mouse_x_gui_open if isGuiOpen else reg_mouse_x,
                    "y": reg_mouse_y_gui_open if isGuiOpen else reg_mouse_y,
                    "dx": delta_x,
                    "dy": delta_y,
                    "scaledX": delta_x * SENS,
                    "scaledY": delta_y * SENS,
                    "buttons": current_buttons,
                    "newButtons": new_buttons
                },
                "keyboard": {
                    "keys": keys_pressed,
                    "newKeys": new_keys_pressed,
                    "chars": chars
                },
                "isGuiOpen": isGuiOpen,
                "isGuiInventory": isGuiInventory,
                "hotbar": current_hotbar,
                "yaw": delta_x * SENS,  # Movimento orizzontale della telecamera
                "pitch": delta_y * SENS,  # Movimento verticale della telecamera
                "xpos": obs.get("xpos", 0.0),
                "ypos": obs.get("ypos", 0.0),
                "zpos": obs.get("zpos", 0.0),
                "tick": tick,
                "milli": current_time,
                "inventory": formatted_inventory,
                "serverTick": server_tick,
                "serverTickDurationMs": server_tick_interval,
                "stats": {}  # Aggiungi logica per raccogliere le statistiche
            }

            # Add the in-game 'ESC' action to the beginning of the action
            action_log.append(registerAction)
            tick += 1
            action = {'ESC': 0, **action}
        
            # Apply the action in the environment
            obs, reward, done, _ = env.step(action)
        
            # Check if the 'q' key is pressed to terminate the loop
            if keys[pygame.K_q]:
                break
        
    except KeyboardInterrupt:
        pass
    finally:    
        # env.render()
        # Cleanup
        out.release()
        # cv2.destroyAllWindows()
        pygame.quit()

    # Save the actions to a JSONL file
    with open(ACTION_LOG_FILE, 'w') as f:
        for action in action_log:
            f.write(json.dumps(action) + '\n')

    cv2.namedWindow('Recorded Video', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(OUTPUT_VIDEO_FILE)

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        # Set the video capture to return grayscale frames
        # cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Recorded Video', frame)
            
            # Adjust the delay for cv2.waitKey() if the playback speed is not correct
            if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()