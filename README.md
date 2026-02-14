# FiaCraft - Progetto per il corso di Fondamenti di Intelligenza Artificiale

## Introduzione
**FiaCraft** è un progetto universitario (corso di Fondamenti di IA) volto ad addestrare agenti su Minecraft utilizzando [MineRL](https://minerl.readthedocs.io/en/latest/) e l'apprendimento per imitazione.<br>
Il nostro team ha lavorato attivamente all'**estensione della libreria originale**, introducendo nuove funzionalità per migliorare l'addestramento e la versatilità degli agenti:

* **📈 Data Augmentation:** Sviluppo di script dedicati per l'aumento sintetico dei dati di addestramento, migliorando la generalizzazione del modello.
* **🧠 Reinforcement Learning Avanzato:** Implementazione di un modulo di apprendimento per rinforzo basato su una **funzione di reward proprietaria**, integrata con meccanismi di feedback umano.
* **🌲 Nuovo Ambiente Custom:** Definizione completa di un nuovo task e rilascio dell'ambiente `(FIA-WoodenPickaxe-v0)`, appositamente ingegnerizzato per testare le nuove capacità acquisite dall'agente.

## ✨ Funzionalità Principali

L'obiettivo centrale del progetto è l'ottimizzazione delle fasi iniziali di gioco (**Early Game**). L'agente è stato addestrato per completare la sequenza di sopravvivenza base massimizzando la velocità di esecuzione e la razionalità decisionale.

### 🛠️ Definizione del Task (Pipeline)
Abbiamo progettato un task sequenziale che richiede all'agente di completare la seguente catena di azioni:

* **🪵 Raccolta:** Individuazione e abbattimento di alberi per l'acquisizione di tronchi di legno di qualsiasi tipo.
* **🪚 Lavorazione:** Trasformazione delle risorse grezze in assi di legno (*planks*).
* **📦 Setup:** Crafting del **Banco da lavoro**, essenziale per sbloccare ricette complesse.
* **⛏️ Strumentazione:** Costruzione finale di un **Piccone di Legno** , completando l'obbiettivo di questo addestramento.

## ⚙️ Installazione Manuale

Se preferisci configurare l'ambiente manualmente, segui questi passaggi.

> ⚠️ **Nota sulla Compatibilità:** Questo progetto è ottimizzato per sistemi **Linux**.
> L'installazione su **Windows** o **macOS** è nota per essere estremamente complessa e instabile (spesso richiede workaround avanzati). Procedi su questi sistemi operativo a tuo rischio.

### 1. Clona la Repository
Scarica il codice sorgente e spostati nella cartella del progetto:
```bash
git clone [https://github.com/MarcoBattag/FiaCraft.git](https://github.com/MarcoBattag/FiaCraft.git)
cd FiaCraft

### 2. Installazione requirements.txt
- **Crea un ambiente virtuale:**<br>
  Puoi usare le versioni di python che vanno dalla 3.8 alla 3.10(consigliato), personalmente abbiamo scelto 3.10<br>
  (assicurati di usare questa versione quando creerai il tuo ambiente):
  ```bash
    sudo apt install python3.10
  ```
  Suggeriamo di utilizzare un ambiente virtuale per isolare le dipendenze del progetto
  ```bash
    python3 -m venv myenv
    source myenv/bin/activate
  ```
  
- **Installa le dipendenze richieste eseguendo:**
  ```bash
    pip install -r requirements.txt
  ```
  
- **Installa MineRL:**<br>
  Tieni conto che MineRL ha bisogno di Java 8:
  ```bash
    sudo apt install openjdk-8-jdk
   ```
  Ora puoi installare MineRL, il modo più semplice per farlo è tramite il repository ufficiale (v1.0.2)
   ```bash
    pip install git+https://github.com/minerllabs/minerl
   ```
  Adesso dovresti avere tutto ciò che ti serve per eseguire il progetto!