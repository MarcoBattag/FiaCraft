import json
import pygame
import argparse

# Impostazioni della finestra
WIDTH, HEIGHT = 1280, 720
BACKGROUND_COLOR = (0, 0, 0)
BALL_COLOR = (255, 0, 0)
BALL_RADIUS = 5
FPS = 30

def extract_coordinates(file_path):
    """Estrae le coordinate x e y dal file JSONL."""
    coordinates = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                x = data["mouse"].get("x", 0)
                y = data["mouse"].get("y", 0)
                coordinates.append((x, y))
            except (json.JSONDecodeError, KeyError):
                continue
    return coordinates

def main(file_path):
    # Estrai le coordinate
    coordinates = extract_coordinates(file_path)
    if not coordinates:
        print("Nessuna coordinata valida trovata nel file.")
        return

    # Inizializza Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mouse Movement Visualization")
    clock = pygame.time.Clock()

    running = True
    index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Pulisci lo schermo
        screen.fill(BACKGROUND_COLOR)

        # Disegna il pallino
        if index < len(coordinates):
            x, y = coordinates[index]
            x = min(max(0, int(x)), WIDTH - 1)  # Mantieni le coordinate nei limiti
            y = min(max(0, int(y)), HEIGHT - 1)
            print(f"Coordinata attuale: x={x}, y={y}")  # Stampa le coordinate
            pygame.draw.circle(screen, BALL_COLOR, (x, y), BALL_RADIUS)
            index += 1
        else:
            print("Fine delle coordinate, chiusura del programma.")
            running = False  # Termina il loop quando tutte le coordinate sono state visualizzate

        # Aggiorna lo schermo
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza il movimento del mouse dal file JSONL.")
    parser.add_argument("file", type=str, help="Percorso al file JSONL contenente le coordinate del mouse.")
    args = parser.parse_args()

    main(args.file)
