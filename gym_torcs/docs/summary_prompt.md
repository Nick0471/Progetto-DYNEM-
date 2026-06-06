# DYNEM Project - TORCS Imitation Learning Summary

Questo file funge da "System Prompt" o riassunto di stato per aggiornare rapidamente qualsiasi assistente IA sul contesto, lo stato attuale e gli obiettivi del progetto.

## 📌 1. In cosa consiste il progetto
Il progetto **DYNEM** mira a sviluppare un agente autonomo in grado di guidare nel simulatore di corse **TORCS** (The Open Racing Car Simulator) utilizzando tecniche di **Imitation Learning** (Behavioral Cloning).
- **Interfaccia**: L'agente comunica con TORCS in Python tramite protocollo UDP (SCR).
- **Metodologia**: Registrazione di dati di guida umana (es. tramite DualShock 4) che includono lo stato della vettura (sensori di bordo pista, velocità, angolazione) e le azioni umane (sterzo, acceleratore, freno). Successivamente, questi dati vengono usati per addestrare modelli di Machine Learning (es. KNN, Reti Neurali) in grado di mappare gli stati della vettura in azioni di guida.
L'obiettivo del progetto è quello di addestrare un modello tramite KNN e di riuscire a compiere un giro di pista su Corkscrew nel miglior tempo possibile.

## ✅ 2. Cosa è stato fatto fino ad ora
1. **Script di Acquisizione Dati**: Sviluppato e funzionante (`manual_control_ds4.py`). Permette di guidare in TORCS e salvare lo stato e le azioni in file CSV a ~50Hz.
2. **Raccolta Dati Iniziale**: È stato raccolto un primo dataset guidando sul circuito **Corkscrew**. La guida si è concentrata sull'ottenere il giro più veloce e pulito possibile.
3. **Pipeline di Preparazione (Step 1)**: Lo script `step1_prepare_data.py` è stato eseguito con successo. Ha pulito i dati, unito i CSV, normalizzato le feature (salvando uno scaler) e generato grafici di Exploratory Data Analysis (EDA).
4. **Analisi Criticità Dati**: Durante l'EDA, si è notata una correlazione molto debole (-0.12) tra l'angolo della vettura rispetto alla pista (`angle`) e l'input di sterzo (`target_steer`). Ciò è dovuto alla natura dei dati raccolti: guida "troppo perfetta" con correzioni di sterzo microscopiche (max ~0.05) e assenza di situazioni di "sbandamento". Nonostante ciò, si è deciso di procedere provvisoriamente con questi dati per testare l'intera pipeline.
5. **Struttura di Training e Testing (Step 2 e 3)**: Sono pronti gli script `step2_train_knn.py` (addestra un modello K-Nearest Neighbors come baseline) e `step3_knn_drive.py` (client che testa il modello addestrato in tempo reale su TORCS).

## 🚀 3. Passaggi futuri e Risoluzione dei Problemi
I prossimi passi consistono nell'addestrare il modello KNN (Step 2) e testarlo in pista (Step 3). Qualora sorgessero errori o l'agente non performasse come sperato (molto probabile data la natura del dataset attuale), ecco le direttive da seguire:

- **Problema: L'agente esce di pista (Compounding Error)**
  - *Causa*: Il modello ha imparato a guidare solo in condizioni perfette. Appena commette un piccolo errore e si avvicina al bordo, non sa come tornare in traiettoria.
  - *Azione Futura*: **Migliorare il Dataset**. Sarà necessario raccogliere "Dati di Recupero" (metodo DAgger). Bisognerà registrare nuovi giri guidando intenzionalmente l'auto vicino ai bordi dell'erba e sbandando leggermente, per poi registrare le forti sterzate correttive per riportarla al centro. Questo insegnerà al modello la relazione tra alti valori di `angle` e forti azioni di `target_steer`.

- **Problema: Modello troppo lento (alta latenza in inferenza)**
  - *Causa*: Il modello KNN deve calcolare le distanze rispetto a tutti i campioni del dataset. Più il dataset cresce (es. 10-15 giri previsti), più i millisecondi richiesti per l'inferenza aumentano, rallentando la reazione su TORCS.

- **Problema: Basso R² o Errore MAE alto nel Training**
  - *Causa*: Pochi dati o sbilanciamento estremo.
  - *Azione Futura*: Raccogliere più giri (obiettivo: 10-15 giri). Valutare tecniche di bilanciamento (sovracampionare le curve o sottocampionare i lunghi rettilinei) se il modello tende ad andare sempre dritto.
