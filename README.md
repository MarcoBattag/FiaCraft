# FiaCraft - Progetto per il corso di Fondamenti di Intelligenza Artificiale

## Introduzione
**FiaCraft** è un progetto universitario (corso di Fondamenti di IA) volto ad addestrare agenti su Minecraft utilizzando [MineRL](https://minerl.readthedocs.io/en/latest/) e l'apprendimento per imitazione.<br>
Il nostro team ha lavorato attivamente all'**estensione della libreria originale**, introducendo nuove funzionalità per migliorare l'addestramento e la versatilità degli agenti:

* **📈 Data Augmentation:** Sviluppo di script dedicati per l'aumento sintetico dei dati di addestramento, migliorando la generalizzazione del modello.
* **🧠 Reinforcement Learning Avanzato:** Implementazione di un modulo di apprendimento per rinforzo basato su una **funzione di reward proprietaria**, integrata con meccanismi di feedback umano.
* **🌲 Nuovo Ambiente Custom:** Definizione completa di un nuovo task e rilascio dell'ambiente `(FIA-WoodenPickaxe-v0)`, appositamente ingegnerizzato per testare le nuove capacità acquisite dall'agente.

## ✨ Funzionalità Principali

L'obiettivo centrale del progetto è l'ottimizzazione delle fasi iniziali di gioco (**Early Game**). L'agente è stato addestrato per completare la sequenza di sopravvivenza base massimizzando la velocità di esecuzione e la razionalità decisionale.

### 🛠️ Definizione del Task
Abbiamo progettato un task sequenziale che richiede all'agente di completare la seguente catena di azioni:

* **🪵 Raccolta:** Individuazione e abbattimento di alberi per l'acquisizione di tronchi di legno di qualsiasi tipo.
* **🪚 Lavorazione:** Trasformazione delle risorse grezze in assi di legno (*planks*).
* **📦 Setup:** Crafting del **Banco da lavoro**, essenziale per sbloccare ricette complesse.
* **⛏️ Strumentazione:** Costruzione finale di un **Piccone di Legno** , completando l'obbiettivo di questo addestramento.

## ⚙️ Installazione Manuale
> ⚠️ **Nota sulla Compatibilità:** Questo progetto è ottimizzato per sistemi **Linux**.
> L'installazione su **Windows** o **macOS** è nota per essere estremamente complessa e instabile (spesso richiede workaround avanzati). Procedi su questi sistemi operativo a tuo rischio.

### 1. Clona la Repository
Scarica il codice sorgente e spostati nella cartella del progetto:
```bash
git clone [https://github.com/MarcoBattag/FiaCraft.git](https://github.com/MarcoBattag/FiaCraft.git)
cd FiaCraft
```
### 2. Creazione ambiente virtuale
**Crea un ambiente virtuale:**<br>
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

### 3.📂 Struttura del progetto
Il progetto include i seguenti file principali:

### 🧠 Training
- **`train.py`**: Script per addestrare gli agenti su vari task utilizzando Behavioural Cloning.
- **`behavioural_cloning.py`**: Implementazione dettagliata del processo di addestramento con gestione dello stato e ottimizzazioni.
- **`rf_learning.py`**: Implementazione dettagliata del processo di addestramento con reward automatiche.
- **`rl_human_feedback.py`**: Implementazione dettagliata del processo di addestramento con reward manuali.
- **`FIAenv.py`**: Implementazione dell'ambiente.

### Testing
tipologie di testing come monitorare l'errore non si sono dimostrati metodi efficaci, siccome l'agente in un dato momento puo eseguire N azioni valide e diverse dal comportamento atteso, il metodo migliore per testare il modello è stato mediante un giudizio umano, sulla base degli oggetti ottenuti e sul quanto le sue azioni appaiano "ragionate". Per questo abbiamo usato il rf_learning usando una combinazione di episodi e passi massimi tenendo sempre monitorato l'inventario dell'agente a fine episodio e utilizzando un sistema di ricompense in base ai blocchi che distruggeva/costruiva
```bash
 python run_agent.py --model data/VPT-models/foundation-model-1x.model --weights data/TRAIN-models/finetuned-1x.weights --env FIA-WoodenPickaxe-v0 --show
```
### 🛠️ Altri File (COTS)
- **`data_loader.py`**: Caricamento dei dataset per il training e il testing.
.

## 🚀Esecuzione del Progetto
### Esecuzione di un modello pre addestrato
Una volta addestrato il tuo modello non ti basterà altro che eseguire `run_agent.py` nel seguente modo, specificando il modello da usare, i pesi da utilizzare e ambiente in cui svolgere il task (FIA-Treechop-v0), apparirà una finestra, e mediante una particolare versione di minecraft integrata in MineRL potrai vedere l'agente agire nel mondo di Minecraft:
```bash
 python run_agent.py --model data/VPT-models/foundation-model-1x.model --weights data/TRAIN-models/finetuned-1x.weights --env FIA-WoodenPickaxe-v0 --show
```
### Training per imitazione
Per addestrare un agente per imitazione del comportamento esegui il seguente file come mostrato.<br>
Durante l'addestramento verra visualizzato sul terminale, e stasmpato su un file excel il valore della funzione di loss. così sarà possibile rendersi conto dell'andamento dell'addestramento in qualsiasi momento:<br>
(assicurati di aver impostato il modello e i pesi desiderati prima di avviare un addestramento, puoi<br>
farlo editando il file `train.py`)
```bash
python behavioural_cloning.py --data-dir data/data-video/iron --in-model data/VPT-models/foundation-model-1x.model --in-weights data/VPT-models/foundation-model-1x.weights --out-weights data/TRAIN-models/finetuned-1x.weights
```

### Training per rinforzo
Come detto inizialmente abbiamo creato una funzione di reward ad hoc per il nostro task che attribuisce punteggi all'agente ogni volta che raccoglie uno di questi items:<br>
(anche in questo caso all'avvio dell'addestramento sarà presente un grafico che mostrerà l'andamento in tempo reale della reward attribuita, oltre l'effettiva finestra di gioco per visualizzare le azioni in tempo reale)
```python
MATERIAL_REWARDS = {
    "log": 2.0,            
    "planks": 4.0,         
    "stick": 6.0,
    "crafting_table": 20.0,
    "wooden_pickaxe": 500.0,
    "dirt": -0.2,
    "sand": -0.2,
    "gravel": -0.2 
}
```
In modo che una volta appresi i comportamenti desiderati l'agente possa specializzarsi sempre di più in merito alla risoluzione del suo task e correggere alcuni comportamenti indesiderati:
```bash
  python rf_learningGpu.py --env FIA-WoodenPickaxe-v0 --model data/VPT-models/foun
dation-model-1x.model --weights data/TRAIN-models/bc_wooden_pickaxe2_best.weights --episodes 20 --max-steps 6000 --show
```

## 📊 Dataset
Il progetto utilizza parzialmente il dataset BASALT di MineRL, sono stati estratti segmenti di alcuni video dal dataset originale in modo che fossero utili al raggiungimento del nostro task.<br>
Per un 40% i dati sono stati generati e specchiati da noi mediante gli appositi script.<br>

### ✂️ Estrazione dati da dataset esistente
Utile ad estrarre i fotogrammi utili al nostro obiettivo dal dataset fornito da OpenAI.<br>
Sfrutta un semplice filtro che analizza il dataset iniziale e ne genera uno nuovo quando il soggetto<br>
del video conserva nell'inventario uno dei seguenti oggetti:<br>
```python
    useful_items = {
        "crafting_table","oak_planks", "birch_planks", "spruce_planks", 
        "jungle_planks", "acacia_planks", "dark_oak_planks",
        "oak_log", "birch_log", "spruce_log", "jungle_log", 
        "acacia_log", "dark_oak_log"
    }
```
Volendo è possibile cambiare il criterio con cui vengono tagliati i video cambiando questi items, per eseguire lo script invece:
```bash
python CutData.py
```
### 🎮 Generazione manuale (.mp4 e JSONL)
Permette all'utente di sviluppare un breve gameplay nel mondo mi Minecraft tramite una finestra MineRL con lo scopo di generare una nuova osservazione e creare nuovi dati di addestramento, ciò è utile nella definizione di un nuovo task per cui non sono presenti dati.
```bash
python manual_recorder.py
```
### 🪞 Mirroring dei dati
Per aumentare la quantità di dati abbiamo deciso di implementare uno script capace di specchiare i video e le azioni correlate ad ogni fotogramma,
non vengono specchiati fotogrammi che potrebbero causare problemi (es. non viene specchiato quando la GUI è aperta)
```bash
python MirrorData.py --input_folder ./path-videos-to-mirror --output_folder ./path-to-save-mirrored-videos
```
### 🧐 Data Quality
Per verificare la qualità dei dati è stato creato lo script `visualize_mouse_movement.py`, ci ha permesso di accertarci che i movimenti del mouse registrati nel JSONL fossero coerenti con gli spostamenti effettivi nei video.
Semplicemente lo script mostra a video le coordinate riportate nel JSONL, eseguibile con:
```bash
python visualize_mouse_movement.py ./path-to-JSONL-file
```
## 📜 Licenza
Questo progetto è rilasciato sotto la licenza MIT. Consulta il file `LICENSE` per maggiori dettagli.

