import pandas as pd
import matplotlib.pyplot as plt

# Leggi il file Excel
file_path = "../../training_loss_log.xlsx"
df = pd.read_excel(file_path)

# Estrarre i dati di interesse
time = df["Time (s)"]  
avg_loss = df["Average Loss"]

# Creare il grafico
plt.figure(figsize=(10, 6))
plt.plot(time, avg_loss, label="Average Loss", color="blue", linewidth=2)

# Aggiungere etichette e titolo
plt.title("Andamento della Loss durante l'Addestramento", fontsize=14)
plt.xlabel("Tempo (secondi)", fontsize=12)
plt.ylabel("Perdita Media (Loss)", fontsize=12)

# Aggiungere una griglia e una legenda
plt.grid(True)
plt.legend()

# Salvare il grafico come immagine
plt.savefig("loss_trend.png")

# Mostrare il grafico
plt.show()
