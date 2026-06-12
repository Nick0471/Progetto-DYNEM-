"""
Script di analisi dei report JSON.
Legge i risultati di step1 e step2 per fornire un feedback testuale
sulla qualità del dataset e sulle metriche del modello KNN.
"""

import os
import sys
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_step1():
    print("=" * 50)
    print("  ANALISI REPORT STEP 1 (Preparazione Dati)")
    print("=" * 50)
    
    path = os.path.join(BASE_DIR, "reports", "report_step1.json")
    if not os.path.exists(path):
        print(f"File non trovato: {path}")
        return False
        
    with open(path, "r") as f:
        data = json.load(f)
        
    righe = data.get("righe_totali", 0)
    print(f"\n[1] Dimensione Dataset: {righe} campioni")
    if righe < 5000:
        print("  \u26A0\uFE0F CRITICO: Il dataset \u00e8 molto piccolo. Il modello avr\u00e0 difficolt\u00e0 ad imparare i pattern di guida.")
    elif righe < 20000:
        print("  \u26A0\uFE0F ATTENZIONE: Il dataset \u00e8 sufficiente, ma avere pi\u00f9 dati migliorerebbe la stabilit\u00e0.")
    else:
        print("  \u2705 OTTIMO: Il dataset \u00e8 bello corposo e offre molti esempi al modello.")
        
    std_target = data.get("std_target", {})
    print("\n[2] Varianza delle azioni (Deviazione Standard):")
    for target, std in std_target.items():
        print(f"  - {target}: {std:.4f}")
        if std < 0.05:
            print(f"      \u26A0\uFE0F {target} ha una deviazione standard molto bassa. Significa che l'azione \u00e8 quasi sempre costante. Potresti aver bisogno di una guida pi\u00f9 dinamica.")
        else:
            print(f"      \u2705 C'\u00e8 una buona variabilit\u00e0 in {target}.")
            
    return True

def analyze_step2():
    print("\n" + "=" * 50)
    print("  ANALISI REPORT STEP 2 (Addestramento KNN)")
    print("=" * 50)
    
    path = os.path.join(BASE_DIR, "reports", "report_step2.json")
    if not os.path.exists(path):
        print(f"File non trovato: {path}")
        return False
        
    with open(path, "r") as f:
        data = json.load(f)
        
    results = data.get("results", {})
    
    # Valutazione R2
    print("\n[1] Valutazione R\u00b2 (Coefficiente di Determinazione):")
    print("    (L'R\u00b2 indica quanto le predizioni del modello si avvicinano al comportamento reale. Max: 1.0)")
    for target, metrics in results.items():
        r2 = metrics.get("r2", 0)
        print(f"  - {target}: R\u00b2 = {r2:.4f}")
        if r2 < 0.5:
            print("      \u274c SCARSO: Il modello fatica enormemente a prevedere questa azione. Manca logica o servono altre feature.")
        elif r2 < 0.7:
            print("      \u26A0\uFE0F ACCETTABILE: Il modello ha imparato i pattern di base ma \u00e8 impreciso.")
        elif r2 < 0.85:
            print("      \u2705 BUONO: Il modello prevede l'azione in maniera piuttosto fedele alla realt\u00e0.")
        else:
            print("      \u2B50 ECCELLENTE: Il modello capisce perfettamente quando e come compiere quest'azione.")

    # Valutazione MAE
    print("\n[2] Valutazione Errore Medio Assoluto (MAE):")
    print("    (Indica di quanto, in media, la predizione si discosta dal valore reale. Pi\u00f9 \u00e8 basso, meglio \u00e8.)")
    for target, metrics in results.items():
        mae = metrics.get("mae", 0)
        print(f"  - {target}: MAE = {mae:.4f}")
        if mae < 0.05:
            print("      \u2705 L'errore medio \u00e8 bassissimo. Ottima precisione!")
        elif mae < 0.15:
            print("      \u2705 L'errore medio \u00e8 contenuto, il comportamento su strada dovrebbe essere fluido.")
        else:
            print("      \u26A0\uFE0F L'errore \u00e8 un po' alto. Potresti notare scatti o imprecisioni alla guida.")

    return True

def main():
    print("\n\U0001F50E ANALISI AUTOMATICA DEI MODELLI DYNEM \U0001F50E")
    step1_ok = analyze_step1()
    step2_ok = analyze_step2()
    
    print("\n" + "=" * 50)
    if step1_ok and step2_ok:
        print("Analisi completata! Leggi i feedback sopra per capire come sta andando il tuo modello.")
    else:
        print("Impossibile completare l'analisi. Assicurati che i report JSON siano stati generati.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
