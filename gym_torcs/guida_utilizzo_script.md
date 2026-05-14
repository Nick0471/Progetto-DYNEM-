# Guida Operativa – Imitation Learning su TORCS

> Questa guida spiega come usare i tre script nell'ordine corretto,
> cosa aspettarsi ad ogni step, e come diagnosticare gli errori più comuni.

---

## Prerequisiti

Prima di iniziare, verifica che questi elementi siano a posto.

### 1. Struttura cartelle attesa

```
gym_torcs/
├── dataset_laps/          ← I tuoi giri CSV (lap_001_..., lap_002_...)
│   ├── lap_001_time_...csv
│   └── lap_002_time_...csv
├── models/                ← Creata automaticamente dallo step 1
├── plots/                 ← Creata automaticamente dallo step 1
├── step1_prepare_data.py
├── step2_train_knn.py
└── step3_knn_drive.py
```

> [!CAUTION]
> Elimina la cartella `dataset/` con i 5 giri Alpine prima di procedere.
> Non viene letta dagli script (leggono solo `dataset_laps/`), ma è un refuso da rimuovere.

### 2. Dipendenze Python

Apri un terminale nella cartella `gym_torcs/` e installa:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Come verificare che siano installate:**
```powershell
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('OK')"
```
Output atteso: `OK`

---

## STEP 1 – Preparazione Dataset (`step1_prepare_data.py`)

### Cosa fa

- Legge **tutti** i file `lap_*.csv` in `dataset_laps/`
- Li unisce in un unico dataset
- Rimuove righe problematiche (auto ferma, fuori pista, NaN)
- Produce statistiche e 3 grafici nella cartella `plots/`
- Salva `models/dataset_clean.csv`, `models/scaler.pkl`, `models/feature_names.pkl`

### Comando

```powershell
cd c:\Users\myria\Desktop\torcs\Progetto-DYNEM-\gym_torcs
python step1_prepare_data.py
```

### Output atteso (esempio con 2 giri)

```
=======================================================
  STEP 1 – Preparazione Dataset
=======================================================

[1/5] Caricamento CSV da dataset_laps/...
  [lap_001_time_01-14-052_...csv] ->  3605 righe
  [lap_002_time_01-10-799_...csv] ->  3457 righe

  Totale righe dopo unione: 7062

[2/5] Pulizia dati...
  Righe rimosse durante pulizia: 108  (1.5%)
  Righe finali nel dataset pulito: 6954

[3/5] Statistiche descrittive...
  ...

[4/5] Generazione grafici EDA...
  Salvato: ...\plots\eda_distributions.png
  Salvato: ...\plots\eda_correlations.png
  Salvato: ...\plots\eda_track_positions.png

[5/5] Normalizzazione e salvataggio scaler...
  Scaler salvato: ...\models\scaler.pkl
  ...

  ✓ STEP 1 COMPLETATO
  Dataset pronto: 6954 campioni, 25 feature
```

### Cosa guardare nei grafici

Apri la cartella `plots/` e controlla:

| Grafico | Cosa verificare |
|---|---|
| `eda_distributions.png` | Lo sterzo dovrebbe essere centrato su 0. Se è fortemente sbilanciato a destra/sinistra, probabilmente il circuito è molto asimmetrico (normale). Il freno dovrebbe avere un picco alto a 0 (rettilineo) |
| `eda_correlations.png` | `angle` e `target_steer` devono avere correlazione alta (>0.7). Se è bassa c'è un problema nei dati |
| `eda_track_positions.png` | La curva `trackPos` deve oscillare attorno a 0. Se vedi spike oltre ±1.3, sono stati rimasti uscite di pista parziali |

### Errori comuni e soluzioni

| Errore | Causa | Soluzione |
|---|---|---|
| `FileNotFoundError: Nessun file lap_*.csv trovato` | La cartella `dataset_laps/` è vuota | Registra almeno un giro con `manual_control_ds4.py` |
| `ModuleNotFoundError: No module named 'seaborn'` | Dipendenze non installate | `pip install seaborn matplotlib` |
| `KeyError: 'track_0'` | Il CSV ha colonne diverse dalla versione attesa | Controlla che `manual_control_ds4.py` sia aggiornato |
| `UnicodeEncodeError` | Terminale Windows senza UTF-8 | Già risolto nello script con `sys.stdout.reconfigure` |
| Poche righe dopo pulizia (<500) | Il giro era troppo corto o quasi tutto fuori pista | Registra giri più puliti |

---

## STEP 2 – Training KNN (`step2_train_knn.py`)

> [!IMPORTANT]
> Esegui sempre step1 prima di step2. Se aggiungi nuovi giri al dataset, riesegui step1 e poi step2.

### Cosa fa

- Carica `models/dataset_clean.csv` e `models/scaler.pkl`
- Riduce l'indice a 1500 centroidi via KMeans (per latenza <1ms)
- Addestra il `KNeighborsRegressor` multi-output (steer, accel, brake)
- Valuta le performance sul 20% di dati tenuti da parte (test set)
- Salva `models/knn_model.pkl`
- Genera `plots/train_predictions.png` e `plots/train_residuals.png`

### Comandi disponibili

```powershell
# Uso base (k=3, già ottimale)
python step2_train_knn.py

# Cerca il k migliore con cross-validation (più lento, utile con più dati)
python step2_train_knn.py --find-k

# Specifica k manualmente
python step2_train_knn.py --k 5

# Solo rivaluta il modello già esistente senza ritraining
python step2_train_knn.py --eval-only

# Con dataset molto grande: aumenta i centroidi
python step2_train_knn.py --max-index 3000
```

### Output atteso

```
── Risultati sul Test Set (80/20 split) ──────────────
  Target              MAE     RMSE       R²
  ──────────────────────────────────────────
  target_steer     0.0172   0.0358   0.9732
  target_accel     0.0600   0.1360   0.9055
  target_brake     0.0131   0.0531   0.8496

  Latenza media inferenza: 0.91 ms/step
  (TORCS step rate: ~20ms → ✓ OK)
```

### Come leggere le metriche

| Metrica | Significato | Valore accettabile |
|---|---|---|
| **MAE steer** | Errore medio assoluto sullo sterzo (range -1..1) | < 0.05 |
| **R² steer** | Quanto bene il KNN spiega la varianza del sterzo | > 0.90 |
| **R² accel** | Qualità previsione acceleratore | > 0.85 |
| **Latenza** | Tempo per predire una singola azione | < 10ms |

### Cosa guardare nei grafici

| Grafico | Cosa verificare |
|---|---|
| `train_predictions.png` | I punti devono essere vicini alla diagonale tratteggiata (y=x). Punti dispersi = modello impreciso |
| `train_residuals.png` | La distribuzione dei residui deve essere centrata su 0 e simmetrica. Un picco spostato indica bias sistematico |

### Errori comuni e soluzioni

| Errore | Causa | Soluzione |
|---|---|---|
| `FileNotFoundError: models/dataset_clean.csv` | Step1 non ancora eseguito | `python step1_prepare_data.py` |
| R² steer < 0.80 | Pochi dati o giri molto incoerenti | Aggiungi giri, usa `--find-k` |
| Latenza > 10ms | Dataset molto grande senza subsampling | Usa `--max-index 1000` |
| `ConvergenceWarning` da KMeans | Max iterazioni raggiunte nel subsampling | Ignorabile, non compromette il risultato |

---

## STEP 3 – Agente KNN in TORCS (`step3_knn_drive.py`)

> [!IMPORTANT]
> TORCS deve essere **già avviato e in modalità Practice/Race** prima di lanciare questo script.
> Il circuito deve essere lo stesso su cui hai registrato i dati (Alpine o altro).

### Sequenza di avvio

```
1. Avvia TORCS
2. Seleziona il circuito (es. alpine-1)
3. Configura come Practice o Quick Race (1 avversario fittizio va bene)
4. Clicca "Accept" e poi START la gara
5. TORCS si mette in pausa aspettando il client
6. Apri un terminale e lancia step3
```

### Comandi disponibili

```powershell
# Base: l'agente KNN controlla tutto
python step3_knn_drive.py

# Con fallback: se esce di pista usa il bot deterministico per recuperare
python step3_knn_drive.py --fallback

# Con telemetria dettagliata ad ogni step
python step3_knn_drive.py --verbose

# Su porta diversa (se hai più istanze TORCS)
python step3_knn_drive.py --port 3002

# Combinazione consigliata per i primi test
python step3_knn_drive.py --fallback --verbose
```

### Output atteso durante la guida

```
  Modello KNN caricato (3 vicini)
  Feature: 25
  Indice ball_tree: pronto.

  Connessione a localhost:3001...
  >>> CONNESSO A TORCS.

  Premi INVIO quando TORCS e' pronto...   ← premi Invio

  step=  100 | spd= 187.3 km/h | pos=+0.123 | ang=+0.012 | gear=5 | st=+0.031 acc=0.980 brk=0.000 [KNN]
  step=  200 | spd= 154.2 km/h | pos=-0.034 | ang=-0.087 | gear=4 | st=-0.215 acc=0.610 brk=0.120 [KNN]
  ...

── Riepilogo sessione ─────────────────────────────
  Step totali : 5000
  Controllo KNN      :  4800  (96.0%)
  Controllo Fallback :   200  ( 4.0%)
```

### Come interpretare il riepilogo

| % Fallback | Valutazione | Cosa fare |
|---|---|---|
| 0–5% | Ottimo – l'agente guida bene | Aggiungi giri per migliorare ancora |
| 5–20% | Discreto – esce di pista raramente | Registra più dati sulle curve difficili |
| 20–50% | Insufficiente – esce spesso | Dataset troppo piccolo o incoerente |
| >50% | Critico – il KNN non guida | Vedi troubleshooting avanzato |

### Cosa osservare in TORCS durante il test

- **Auto rimane in pista?** → KNN funziona
- **Sterzo nervoso/oscillante?** → k troppo basso, prova `--k 7`
- **Auto non frena?** → controllare la distribuzione del freno in step1 (forse pochi dati con frenata)
- **Auto va in retromarcia?** → trackPos molto negativo al via, aspetta che si stabilizzi
- **Connessione non avviene?** → TORCS non è in stato "waiting for client", riavvia la gara

### Errori comuni e soluzioni

| Errore | Causa | Soluzione |
|---|---|---|
| `FileNotFoundError: models/knn_model.pkl` | Step2 non eseguito | `python step2_train_knn.py` |
| `socket.timeout` ripetuto | TORCS non è avviato o non in attesa client | Avvia TORCS e metti in pausa sulla schermata di gara |
| `ConnectionResetError` | TORCS è andato in crash | Riavvia TORCS |
| Auto non si muove | Gear bloccato su 0 | Riavvia la gara in TORCS |
| `ModuleNotFoundError: No module named 'numpy'` | Ambiente Python sbagliato | Usa lo stesso Python in cui hai installato le dipendenze |

---

## Workflow Completo (da zero a agente guidante)

```
PRIMA VOLTA                        AGGIORNAMENTO DATASET
─────────────────                  ─────────────────────
1. Registra giri                   1. Registra nuovi giri
   manual_control_ds4.py              manual_control_ds4.py

2. Prepara dataset                 2. Ri-prepara dataset
   python step1_prepare_data.py       python step1_prepare_data.py

3. Addestra KNN                    3. Ri-addestra KNN
   python step2_train_knn.py          python step2_train_knn.py

4. Avvia TORCS                     4. Avvia TORCS

5. Testa agente                    5. Testa agente migliorato
   python step3_knn_drive.py          python step3_knn_drive.py
   --fallback --verbose               --fallback --verbose
```

---

## Diagnostica Avanzata

### Il modello sembra buono offline ma va male online

Questo è il **compounding error** dell'imitation learning puro:
il modello non ha mai visto stati "di recupero" (es. auto un po' fuori traiettoria),
quindi piccoli errori si accumulano.

**Soluzione a lungo termine:** raccogliere dati anche da posizioni
di recupero (guidare intenzionalmente vicino ai bordi e correggere).

### Come confrontare due modelli

```powershell
# Modello A: k=3
python step2_train_knn.py --k 3
# Nota l'R² e la latenza, poi rinomina il modello
rename models\knn_model.pkl models\knn_k3.pkl

# Modello B: k=7
python step2_train_knn.py --k 7
rename models\knn_model.pkl models\knn_k7.pkl

# Per testare uno specifico: rinominalo in knn_model.pkl prima di step3
copy models\knn_k3.pkl models\knn_model.pkl
python step3_knn_drive.py --fallback
```

### Verificare cosa vede il modello in tempo reale

Aggiungi `--verbose` e osserva la colonna `pos` (trackPos).
Se oscilla rapidamente tra valori positivi e negativi, il modello
sta "correggendo" in modo instabile → aumenta k o aggiungi dati.
