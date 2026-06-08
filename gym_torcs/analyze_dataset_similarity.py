import os
import sys
import glob
import json
import re
import argparse
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

def parse_time_from_filename(filename: str) -> float:
    """Estrae il tempo dal nome del file e lo converte in secondi decimali."""
    # Esempio: lap_001_time_01-09-563_20260528_173959.csv
    match = re.search(r'time_(\d+)-(\d+)-(\d+)', filename)
    if match:
        mins = int(match.group(1))
        secs = int(match.group(2))
        mils = int(match.group(3))
        return (mins * 60) + secs + (mils / 1000.0)
    return None

def is_lap_clean(filepath: str) -> bool:
    """Verifica che il giro non contenga fuoripista (trackPos > 1.3)."""
    try:
        df = pd.read_csv(filepath)
        if len(df) < 100:
            return False
        # Controlliamo se c'è almeno un frame in cui si è usciti di pista
        if (df['trackPos'].abs() > 1.3).any():
            return False
        return True
    except Exception as e:
        print(f"Errore nella lettura di {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Analisi e Pulizia Giri per Tempo")
    parser.add_argument("--tolerance", type=float, default=2.0, 
                        help="Tolleranza in secondi (default: 2.0)")
    parser.add_argument("--target-time", type=float, default=None, 
                        help="Tempo target in secondi (es. 70.0 per 1:10). Se specificato, cerca giri entro +/- tolerance.")
    parser.add_argument("--delete", action="store_true", 
                        help="Elimina fisicamente i file dei giri scartati")
    args = parser.parse_args()

    print("=" * 60)
    print("  ANALISI TEMPI SUL GIRO E PULIZIA DATASET")
    print("=" * 60)
    
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    print(f"Trovati {len(files)} file CSV nel dataset.\n")
    
    if not files:
        print("Nessun file trovato.")
        return

    laps_info = []
    
    print("Analisi dei file in corso (estrazione tempi e controllo fuoripista)...")
    for f in files:
        filename = os.path.basename(f)
        lap_time = parse_time_from_filename(filename)
        
        if lap_time is None:
            continue
            
        clean = is_lap_clean(f)
        
        laps_info.append({
            "filepath": f,
            "filename": filename,
            "time": lap_time,
            "clean": clean
        })

    # Trova il giro di riferimento
    if args.target_time is not None:
        baseline_time = args.target_time
        fastest_lap_name = "Target Personalizzato"
        print(f"\n[+] Giro di riferimento: {fastest_lap_name}")
        print(f"    Tempo impostato: {baseline_time:.3f} sec")
        print(f"    Range accettato: da {baseline_time - args.tolerance:.3f}s a {baseline_time + args.tolerance:.3f}s (±{args.tolerance}s)")
    else:
        clean_laps = [lap for lap in laps_info if lap["clean"]]
        if not clean_laps:
            print("ERRORE: Nessun giro 'pulito' (senza fuoripista) trovato nel dataset!")
            return
            
        # Usa la MEDIA dei tempi invece del giro più veloce
        import numpy as np
        baseline_time = float(np.mean([lap["time"] for lap in clean_laps]))
        fastest_lap_name = "Media dei Tempi"
        
        print(f"\n[+] Giro di riferimento (Media matematica): {fastest_lap_name}")
        print(f"    Tempo medio calcolato: {baseline_time:.3f} sec")
        print(f"    Range accettato: da {baseline_time - args.tolerance:.3f}s a {baseline_time + args.tolerance:.3f}s (±{args.tolerance}s)")
    
    # Classificazione
    kept_laps = []
    discarded_laps = []
    
    for lap in laps_info:
        if args.target_time is not None:
            # Accetta nel range [target - tolerance, target + tolerance]
            if lap["clean"] and abs(lap["time"] - baseline_time) <= args.tolerance:
                kept_laps.append(lap)
            else:
                reason = "Fuoripista rilevato" if not lap["clean"] else f"Fuori target di {abs(lap['time'] - baseline_time):.3f}s"
                lap["reason"] = reason
                discarded_laps.append(lap)
        else:
            # Accetta nel range [media - tolerance, media + tolerance]
            if lap["clean"] and abs(lap["time"] - baseline_time) <= args.tolerance:
                kept_laps.append(lap)
            else:
                diff = lap['time'] - baseline_time
                reason = "Fuoripista rilevato" if not lap["clean"] else f"Fuori media ({'+' if diff>0 else ''}{diff:.3f}s)"
                lap["reason"] = reason
                discarded_laps.append(lap)
            
    print("\n── RISULTATI ──────────────────────────────────────────")
    print(f"Giri BUONI mantenuti : {len(kept_laps)}")
    print(f"Giri SCARTATI        : {len(discarded_laps)}")
    
    if discarded_laps:
        print("\nDettaglio giri scartati:")
        for lap in discarded_laps:
            print(f"  - {lap['filename']}  [{lap['time']:.3f}s] -> {lap['reason']}")
            
    # Generazione Report
    report = {
        "baseline_time_sec": baseline_time,
        "reference": fastest_lap_name,
        "tolerance_sec": args.tolerance,
        "total_laps": len(laps_info),
        "kept_laps": len(kept_laps),
        "discarded_laps": len(discarded_laps),
        "discarded_details": [{"file": l["filename"], "reason": l["reason"]} for l in discarded_laps]
    }
    
    report_path = os.path.join(REPORTS_DIR, "lap_times_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nReport completo salvato in: {report_path}")
    
    # Eliminazione
    if args.delete and discarded_laps:
        print(f"\n[!] ELIMINAZIONE IN CORSO di {len(discarded_laps)} file...")
        deleted_count = 0
        for lap in discarded_laps:
            try:
                os.remove(lap["filepath"])
                deleted_count += 1
            except Exception as e:
                print(f"Errore nell'eliminazione di {lap['filename']}: {e}")
        print(f"Eliminati con successo {deleted_count} file.")
    elif discarded_laps and not args.delete:
        print("\n[i] Nessun file è stato eliminato. Usa '--delete' per cancellare fisicamente i file scartati.")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
