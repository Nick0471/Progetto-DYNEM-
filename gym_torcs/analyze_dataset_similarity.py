import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from itertools import combinations

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

def extract_lap_profile(filepath, num_points=200):
    """
    Estrae il profilo del giro e lo ricampiona a una lunghezza fissa 
    per permettere il confronto tra giri che hanno una durata (numero di step) diversa.
    Usiamo 'trackPos' e 'speedX' come firma del giro.
    """
    df = pd.read_csv(filepath)
    if len(df) < 100:
        return None  # Salta giri troppo brevi
        
    # Estraiamo le colonne chiave
    track_pos = df['trackPos'].values
    speed_x = df['speedX'].values
    
    # Creiamo un asse temporale normalizzato da 0 a 1
    x_old = np.linspace(0, 1, len(df))
    x_new = np.linspace(0, 1, num_points)
    
    # Interpolazione per portare tutti i giri a `num_points` lunghezze
    track_pos_resampled = np.interp(x_new, x_old, track_pos)
    speed_x_resampled = np.interp(x_new, x_old, speed_x)
    
    # Concateniamo i due array per avere una singola "firma" per il giro
    # Normalizziamo le feature per far s\u00ec che abbiano peso simile
    tp_norm = (track_pos_resampled - np.mean(track_pos_resampled)) / (np.std(track_pos_resampled) + 1e-6)
    sp_norm = (speed_x_resampled - np.mean(speed_x_resampled)) / (np.std(speed_x_resampled) + 1e-6)
    
    profile = np.concatenate([tp_norm, sp_norm])
    return profile

def main():
    print("=" * 60)
    print("  ANALISI SIMILARIT\u00c0 DATASET")
    print("=" * 60)
    
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    print(f"Trovati {len(files)} file CSV nel dataset.\n")
    
    profiles = {}
    
    # 1. Estrazione dei profili
    print("Estrazione profili in corso...")
    for f in files:
        filename = os.path.basename(f)
        prof = extract_lap_profile(f)
        if prof is not None:
            profiles[filename] = prof
            
    # 2. Calcolo similarit\u00e0 (distanza a coppie)
    print("Calcolo similarit\u00e0 tra i giri...")
    filenames = list(profiles.keys())
    similar_pairs = []
    
    # Soglia empirica: due giri sono considerati "molto simili" se la distanza
    # Euclidea normalizzata tra le loro firme \u00e8 al di sotto di questo valore.
    THRESHOLD = 15.0 
    
    for f1, f2 in combinations(filenames, 2):
        dist = euclidean(profiles[f1], profiles[f2])
        
        # Trasformiamo la distanza in una "percentuale" di similarit\u00e0 approssimativa
        # (E' un valore indicativo, basato sulla dimensione dell'array)
        max_dist = 40.0 
        sim_score = max(0, 100 * (1 - (dist / max_dist)))
        
        if sim_score > 90.0:  # Pi\u00f9 del 90% di similarit\u00e0
            similar_pairs.append({
                "file_1": f1,
                "file_2": f2,
                "similarita_perc": round(sim_score, 2),
                "distanza": round(dist, 4)
            })
            
    # Ordiniamo i risultati per similarit\u00e0 (dal pi\u00f9 simile al meno)
    similar_pairs = sorted(similar_pairs, key=lambda x: x["similarita_perc"], reverse=True)
    
    # 3. Generazione Report
    report = {
        "info_generali": {
            "csv_totali_analizzati": len(profiles),
            "csv_troppo_corti_scartati": len(files) - len(profiles)
        },
        "giri_simili_trovati": len(similar_pairs),
        "dettagli_similarita": similar_pairs
    }
    
    report_path = os.path.join(REPORTS_DIR, "dataset_similarity_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[!] Trovate {len(similar_pairs)} coppie di giri estremamente simili (>90%).")
    print(f"Report completo salvato in: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
