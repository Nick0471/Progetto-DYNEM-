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
        if (df['trackPos'].abs() > 1.3).any():
            return False
        return True
    except Exception as e:
        print(f"Errore nella lettura di {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Analisi e Pulizia Giri per Tempo")
    parser.add_argument("--tolerance", type=float, default=2.0, 
                        help="Tolleranza in sec (default: 2.0). Usata se non si usa --top.")
    parser.add_argument("--target-time", type=float, default=None, 
                        help="Tempo target in sec. Usato se non si usa --top.")
    parser.add_argument("--top", type=int, default=None, 
                        help="Mantieni solo i N giri puliti più veloci (es. 15)")
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

    kept_laps = []
    discarded_laps = []

    # ─────────────────────────────────────────────
    # NUOVA LOGICA: MODALITÀ TOP N
    # ─────────────────────────────────────────────
    if args.top is not None:
        print(f"\n[+] Modalità Classifica: Estrazione dei TOP {args.top} giri più veloci")
        
        # Filtriamo solo i giri puliti e li ordiniamo per tempo (dal minore al maggiore)
        clean_laps = [lap for lap in laps_info if lap["clean"]]
        clean_laps.sort(key=lambda x: x["time"])
        
        # Prendiamo i primi N
        kept_laps = clean_laps[:args.top]
        kept_filenames = {lap["filename"] for lap in kept_laps}
        
        if kept_laps:
            print(f"    Giro più veloce tra i top: {kept_laps[0]['time']:.3f}s")
            print(f"    Giro più lento tra i top:  {kept_laps[-1]['time']:.3f}s")
        else:
            print("    ERRORE: Nessun giro pulito trovato da inserire in classifica.")

        # Inseriamo tutti gli altri nei discarded
        for lap in laps_info:
            if lap["filename"] not in kept_filenames:
                if not lap["clean"]:
                    lap["reason"] = "Fuoripista rilevato"
                else:
                    lap["reason"] = f"Non rientra nella Top {args.top}"
                discarded_laps.append(lap)
                
    # ─────────────────────────────────────────────
    # LOGICA ORIGINALE (MEDIA O TARGET TIME)
    # ─────────────────────────────────────────────
    else:
        if args.target_time is not None:
            baseline_time = args.target_time
            fastest_lap_name = "Target Personalizzato"
        else:
            clean_laps = [lap for lap in laps_info if lap["clean"]]
            if not clean_laps:
                print("ERRORE: Nessun giro 'pulito' trovato nel dataset!")
                return
            import numpy as np
            baseline_time = float(np.mean([lap["time"] for lap in clean_laps]))
            fastest_lap_name = "Media dei Tempi"
            
        print(f"\n[+] Giro di riferimento: {fastest_lap_name}")
        print(f"    Tempo base: {baseline_time:.3f} sec | Tolleranza: ±{args.tolerance}s")
        
        for lap in laps_info:
            if lap["clean"] and abs(lap["time"] - baseline_time) <= args.tolerance:
                kept_laps.append(lap)
            else:
                diff = lap['time'] - baseline_time
                reason = "Fuoripista rilevato" if not lap["clean"] else f"Fuori limite ({'+' if diff>0 else ''}{diff:.3f}s)"
                lap["reason"] = reason
                discarded_laps.append(lap)

    # ─────────────────────────────────────────────
    # OUTPUT E REPORT
    # ─────────────────────────────────────────────
    print("\n── RISULTATI ──────────────────────────────────────────")
    print(f"Giri BUONI mantenuti : {len(kept_laps)}")
    print(f"Giri SCARTATI        : {len(discarded_laps)}")
    
    if discarded_laps:
        print("\nDettaglio giri scartati:")
        for lap in discarded_laps:
            print(f"  - {lap['filename']}  [{lap['time']:.3f}s] -> {lap['reason']}")
            
    report = {
        "mode": f"Top {args.top}" if args.top else "Tolerance",
        "total_laps": len(laps_info),
        "kept_laps": len(kept_laps),
        "discarded_laps": len(discarded_laps),
        "discarded_details": [{"file": l["filename"], "reason": l["reason"]} for l in discarded_laps]
    }
    
    report_path = os.path.join(REPORTS_DIR, "lap_times_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nReport completo salvato in: {report_path}")
    
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