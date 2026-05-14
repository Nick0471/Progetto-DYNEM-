# Analisi Progetto DYNEM – Imitation Learning su TORCS

## 📌 Cos'è il progetto

Il progetto utilizza **TORCS** (The Open Racing Car Simulator) con interfaccia Python via protocollo SCR (UDP).  
Hai già costruito un sistema di raccolta dati **manuale tramite joypad DualShock 4** e hai registrato giri di guida che salvano telemetria + azioni del guidatore in file CSV.

---

## 📂 Struttura del Progetto

| File/Directory | Ruolo |
|---|---|
| `manual_control_ds4.py` | **Script di raccolta dati** – Guidi con il joypad e registra stato+azione ogni ~20ms |
| `dataset_laps/` | **Dataset raccolto** – 2 giri CSV (giro 1: ~74s, giro 2: ~70s) |
| `dataset/Alpine2_5laps_02_09_01.csv` | Dataset più vecchio (5 giri, ~9MB) |
| `gym_torcs.py` | Wrapper OpenAI Gym per TORCS |
| `torcs_jm_par_modulare.py` | Bot deterministico con logica modulare (baseline) |
| `sample_agent.py` | Struttura agente base (attualmente random) |
| `jmcncarai.py` / `snakeoil3_gym.py` | Client UDP per TORCS |

---

## 🗂️ Struttura del Dataset

Ogni riga del CSV è uno **snapshot** dello stato della vettura + azione umana al momento t.

**35 colonne totali:**

| Gruppo | Colonne | Descrizione |
|---|---|---|
| **Target (label)** | `target_steer`, `target_accel`, `target_brake`, `target_gear` | Azione del guidatore umano |
| **Velocità** | `speedX`, `speedY`, `speedZ` | Velocità longitudinale/laterale/verticale (km/h) |
| **Assetto** | `angle`, `trackPos` | Angolo rispetto alla traiettoria, posizione trasversale [-1,1] |
| **Sensori pista** | `track_0` … `track_18` | 19 raggi laser (distanza dai bordi, angoli da -45° a +45°) |
| **Powertrain** | `gear`, `rpm` | Marcia attuale e giri motore |
| **Ruote** | `wheelSpinVel_0…3` | Velocità angolare ruote (anti-slittamento) |

**Dimensioni:**
- `lap_001`: **3.605 righe** (~74 secondi @ ~49 step/s)
- `lap_002`: ~**3.450 righe** stimate
- Dataset vecchio: ~**30.000+ righe** (5 giri)

---

## 🎯 Prossimi Passi per l'Imitation Learning

### FASE 1 – Preparazione Dati ✅ (già fatto in parte)
1. **Unire tutti i CSV** in un unico DataFrame  
2. **Normalizzare i feature** (StandardScaler o MinMaxScaler)  
3. **Scegliere le feature di input** – le migliori per la guida:
   - `angle`, `trackPos`, `speedX`, `track_0..18` (→ 22 feature totali)
4. **Scegliere i target di output** – steer/accel/brake (trattarli separatamente o multi-output)
5. **Bilanciare il dataset** – i dati in rettilineo sono molto più frequenti delle curve

### FASE 2 – Scelta del Modello

#### ✅ KNN – È possibile? **SÌ, ma con limitazioni**

| Aspetto | Dettaglio |
|---|---|
| **Funziona?** | Sì, il KNN è un ottimo punto di partenza per behavior cloning |
| **Vantaggio** | Zero training, interpretabile, buono per dataset piccoli |
| **Problema 1** | **Lento in inferenza** – con 30k+ campioni e 22 feature, ogni query richiede una distanza da tutti i punti |
| **Problema 2** | **Maledizione della dimensionalità** – con 19+ sensori di pista la distanza Euclidea perde significato |
| **Problema 3** | Non generalizza bene a situazioni mai viste |
| **Soluzione** | Ridurre le feature (PCA o selezione manuale) + usare `KD-Tree` o `Ball-Tree` per velocizzare |

#### 🔵 Algoritmi alternativi (in ordine di complessità crescente)

| Modello | Pro | Contro |
|---|---|---|
| **KNN** (k=5..15) | Semplice, nessun training | Lento, dimensionalità |
| **Random Forest** | Robusto, feature importance | Multi-output richiede wrapper |
| **MLP / Rete Neurale** | Generalizza bene, veloce in inferenza | Richiede più dati e tuning |
| **LSTM / GRU** | Sfrutta la sequenza temporale | Più complesso da implementare |

> **Raccomandazione:** Inizia con KNN come baseline, poi confronta con un MLP a 2-3 layer.

---

## 🛠️ Piano Implementativo Consigliato

### Step 1 – Script di Training KNN

```python
# knn_agent_train.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Carica dati
df = pd.read_csv("dataset_laps/lap_001_time_01-14-052_20260514_092327.csv")
# Aggiungi altri CSV: df = pd.concat([df, pd.read_csv("lap_002...")], ignore_index=True)

# 2. Feature selection
FEATURES = ['angle', 'trackPos', 'speedX', 
            'track_0','track_1','track_2','track_3','track_4',
            'track_5','track_6','track_7','track_8','track_9',
            'track_10','track_11','track_12','track_13','track_14',
            'track_15','track_16','track_17','track_18']
TARGETS = ['target_steer', 'target_accel', 'target_brake']

X = df[FEATURES].values
y = df[TARGETS].values

# 3. Split e normalizzazione
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Training KNN
knn = KNeighborsRegressor(n_neighbors=7, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
knn.fit(X_train, y_train)

# 5. Valutazione
y_pred = knn.predict(X_test)
for i, t in enumerate(TARGETS):
    print(f"{t}: MAE = {mean_absolute_error(y_test[:,i], y_pred[:,i]):.4f}")

# 6. Salva modello
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

### Step 2 – Agente KNN per TORCS

```python
# knn_agent_drive.py
import joblib
import numpy as np

class KNNAgent:
    def __init__(self):
        self.knn = joblib.load("knn_model.pkl")
        self.scaler = joblib.load("scaler.pkl")
        self.FEATURES = ['angle', 'trackPos', 'speedX', 
                         *[f'track_{i}' for i in range(19)]]

    def act(self, state_dict):
        x = np.array([[state_dict.get(f, 0) for f in self.FEATURES]])
        x = self.scaler.transform(x)
        pred = self.knn.predict(x)[0]
        return {
            'steer': float(np.clip(pred[0], -1, 1)),
            'accel': float(np.clip(pred[1], 0, 1)),
            'brake': float(np.clip(pred[2], 0, 1))
        }
```

### Step 3 – Integrazione con TORCS (da scrivere)

Collegare `KNNAgent` al loop UDP esistente in `torcs_jm_par_modulare.py`, sostituendo `drive_modular()` con `knn_agent.act(S)`.

---

## ⚠️ Problemi Noti da Risolvere

| Problema | Causa | Soluzione |
|---|---|---|
| **Pochi dati** | 2 soli giri (~7k righe) | Registra almeno 10-15 giri |
| **Dataset sbilanciato** | Rettilineo >> curve | Oversampling curve o pesatura campioni |
| **Latenza KNN** | ~30k punti × 22 feature | Usa `ball_tree` + riduci feature con PCA |
| **Compounding error** | Piccoli errori si accumulano | Tecnica DAgger (registrare anche dati di recovery) |
| **gear non imparato** | Marce dipendono da rpm/velocità | Usa la logica automatica del bot, non imparare dal KNN |

---

## 📊 Metriche di Valutazione

- **Offline:** MAE su steer/accel/brake sul test set
- **Online (su TORCS):** Tempo per giro, distanza percorsa senza uscire di pista, numero di uscite di pista

---

## 🗺️ Roadmap Consigliata

```
[FATTO]  ✅ Raccolta dati manuale (joypad + CSV)
[PROSSIMO] 1️⃣  Unire + analizzare dataset (EDA, distribuzioni)
           2️⃣  Implementare baseline KNN e valutarla offline
           3️⃣  Integrare KNN nel loop TORCS e testare online
           4️⃣  Raccogliere più dati (specialmente curve e recovery)
           5️⃣  Confrontare KNN con MLP per migliorare le prestazioni
           6️⃣  (Opzionale) DAgger per ridurre compounding error
```
