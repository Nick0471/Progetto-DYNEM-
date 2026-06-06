import os
import sys
import glob
import json
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def augment_dataframe(df, noise_level=0.01):
    """
    SCHELETRO FUNZIONE: Data Augmentation.
    Aggiunge un lievissimo rumore casuale alle feature principali per
    simulare un giro 'leggermente' diverso.
    """
    df_aug = df.copy()
    
    # Aggiungi rumore gaussiano alla posizione in pista (simula traiettorie non perfette)
    if 'trackPos' in df_aug.columns:
        noise = np.random.normal(0, noise_level, len(df_aug))
        df_aug['trackPos'] = df_aug['trackPos'] + noise
        
    # Aggiungi rumore allo sterzo e velocità
    if 'speedX' in df_aug.columns:
        noise = np.random.normal(0, noise_level * 50, len(df_aug)) # Rumore su km/h
        df_aug['speedX'] = df_aug['speedX'] + noise
        
    return df_aug

def main():
    print("=" * 60)
    print("  GESTIONE DATASET (AUGMENTATION / CANCELLAZIONE)")
    print("=" * 60)
    
    report_path = os.path.join(REPORTS_DIR, "dataset_similarity_report.json")
    if not os.path.exists(report_path):
        print("Errore: Il report di similarit\u00e0 non esiste. Esegui prima analyze_dataset_similarity.py")
        return
        
    with open(report_path, "r") as f:
        report = json.load(f)
        
    pairs = report.get("dettagli_similarita", [])
    if not pairs:
        print("Non ci sono CSV simili da processare.")
        return
        
    # Raccogli tutti i file ridondanti (il "file_2" di ogni coppia)
    files_to_process = set([p["file_2"] for p in pairs])
    print(f"Identificati {len(files_to_process)} file potenzialmente ridondanti.")
    
    # LOGICA: MODIFICARE (Consigliata) vs ELIMINARE (Sconsigliata)
    # E' sempre meglio applicare la Data Augmentation piuttosto che eliminare, 
    # perch\u00e8 le differenze millimetriche aiutano la regolarizzazione del KNN.
    
    for filename in files_to_process:
        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath):
            continue
            
        print(f"Processando {filename}...")
        
        # OPZIONE 1: ELIMINAZIONE
        # os.remove(filepath)
        # print(f"  -> Eliminato.")
        
        # OPZIONE 2: MODIFICA (DATA AUGMENTATION)
        df = pd.read_csv(filepath)
        df_augmented = augment_dataframe(df, noise_level=0.015)
        
        # Salviamo sovrascrivendo, o creando un nuovo file
        aug_filename = "augmented_" + filename
        aug_filepath = os.path.join(DATASET_DIR, aug_filename)
        df_augmented.to_csv(aug_filepath, index=False)
        print(f"  -> Modificato e salvato come {aug_filename}")
        
        # OPZIONALE: Se crei un file aumentato, puoi decidere di cancellare l'originale
        # os.remove(filepath)

    print("\nProcesso completato.")

if __name__ == "__main__":
    main()
