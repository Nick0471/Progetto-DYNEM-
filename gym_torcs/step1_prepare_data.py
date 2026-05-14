"""
STEP 1 - Preparazione e Analisi Esplorativa del Dataset (EDA)
=============================================================
Legge tutti i CSV presenti in dataset_laps/, li unisce, esegue
un'analisi statistica e salva il dataset pulito + normalizzato
pronti per il training.

Output generati:
  - models/dataset_merged.csv      -> dati grezzi uniti
  - models/dataset_clean.csv       -> dati puliti e filtrati
  - models/scaler.pkl              -> scaler sklearn (StandardScaler)
  - models/feature_names.pkl       -> lista feature di input usate
  - plots/eda_distributions.png    -> istogrammi target
  - plots/eda_correlations.png     -> heatmap correlazioni
  - plots/eda_track_positions.png  -> traiettoria percorsa

Uso:
  python step1_prepare_data.py
"""

import os
import sys
import glob
import pickle

# Forza stdout UTF-8 su Windows per evitare errori di encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend non-interattivo (nessuna finestra)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# Feature di input per il modello (22 sensori)
FEATURE_COLS = [
    "angle",
    "trackPos",
    "speedX",
    "speedY",
    "speedZ",
    "rpm",
    "track_0",  "track_1",  "track_2",  "track_3",  "track_4",
    "track_5",  "track_6",  "track_7",  "track_8",  "track_9",
    "track_10", "track_11", "track_12", "track_13", "track_14",
    "track_15", "track_16", "track_17", "track_18",
]

# Variabili target (azioni del guidatore)
TARGET_COLS = ["target_steer", "target_accel", "target_brake"]


# ─────────────────────────────────────────────
# 1. CARICAMENTO E UNIONE DEI CSV
# ─────────────────────────────────────────────
def load_all_laps(folder: str) -> pd.DataFrame:
    """Carica tutti i CSV di giro presenti nella cartella e li unisce."""
    files = sorted(glob.glob(os.path.join(folder, "lap_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"Nessun file lap_*.csv trovato in: {folder}\n"
            "Esegui prima manual_control_ds4.py per registrare i giri."
        )

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df["_source_file"] = os.path.basename(fp)   # traccia l'origine
        frames.append(df)
        print(f"  [{os.path.basename(fp)}] -> {len(df):>5} righe")

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Totale righe dopo unione: {len(merged)}")
    return merged


# ─────────────────────────────────────────────
# 2. PULIZIA DEI DATI
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove righe problematiche:
    - NaN su qualsiasi colonna usata
    - speedX <= 0 (auto ferma o in retromarcia all'inizio giro)
    - trackPos fuori range [-1.3, 1.3] (remnanti di uscite di pista)
    """
    n_start = len(df)

    # Rimuovi NaN
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)

    # Rimuovi frame con auto ferma (i primi istanti del giro)
    df = df[df["speedX"] > 1.0]

    # Rimuovi eventuali frame fuori pista rimasti
    df = df[df["trackPos"].abs() <= 1.3]

    # Reset indice
    df = df.reset_index(drop=True)

    n_removed = n_start - len(df)
    print(f"  Righe rimosse durante pulizia: {n_removed}  ({n_removed/n_start*100:.1f}%)")
    print(f"  Righe finali nel dataset pulito: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 3. STATISTICHE DESCRITTIVE
# ─────────────────────────────────────────────
def print_stats(df: pd.DataFrame):
    print("\n── Statistiche target ────────────────────────────────")
    print(df[TARGET_COLS].describe().round(4).to_string())

    print("\n── Statistiche feature principali ────────────────────")
    print(df[["angle", "trackPos", "speedX", "rpm"]].describe().round(2).to_string())

    # Distribuzione marce (informativa, non usata come target)
    print("\n── Distribuzione marce registrate ────────────────────")
    if "target_gear" in df.columns:
        print(df["target_gear"].value_counts().sort_index().to_string())

    # Percentuale frame con frenata > 0
    brake_pct = (df["target_brake"] > 0.05).mean() * 100
    steer_straight_pct = (df["target_steer"].abs() < 0.05).mean() * 100
    print(f"\n  Frame con frenata attiva  : {brake_pct:.1f}%")
    print(f"  Frame in rettilineo (|steer|<0.05): {steer_straight_pct:.1f}%")


# ─────────────────────────────────────────────
# 4. PLOT EDA
# ─────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Distribuzione delle Azioni del Guidatore", fontsize=13, fontweight="bold")

    colors = ["#3b82f6", "#22c55e", "#ef4444"]
    labels = ["Sterzo (steer)", "Acceleratore (accel)", "Freno (brake)"]

    for ax, col, color, label in zip(axes, TARGET_COLS, colors, labels):
        ax.hist(df[col], bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.set_title(label)
        ax.set_xlabel("Valore [-1..1 / 0..1]")
        ax.set_ylabel("Frequenza")
        ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1, label=f"media={df[col].mean():.3f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "eda_distributions.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvato: {out}")


def plot_correlations(df: pd.DataFrame):
    key_cols = ["angle", "trackPos", "speedX", "rpm"] + TARGET_COLS
    corr = df[key_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Correlazioni tra Sensori e Azioni", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "eda_correlations.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvato: {out}")


def plot_track_positions(df: pd.DataFrame):
    """Mappa approssimata della traiettoria: usa trackPos e speedX come proxy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Analisi Traiettoria e Velocità", fontsize=12, fontweight="bold")

    # trackPos nel tempo
    axes[0].plot(df.index, df["trackPos"], color="#6366f1", linewidth=0.4, alpha=0.8)
    axes[0].axhline(0, color="green", linewidth=1, linestyle="--", label="centro pista")
    axes[0].axhline( 1.0, color="orange", linewidth=0.8, linestyle=":", label="cordolo")
    axes[0].axhline(-1.0, color="orange", linewidth=0.8, linestyle=":")
    axes[0].set_title("Posizione Trasversale (trackPos)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("trackPos")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Profilo velocità
    axes[1].plot(df.index, df["speedX"], color="#f59e0b", linewidth=0.5, alpha=0.8)
    axes[1].set_title("Profilo Velocità (speedX)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("km/h")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "eda_track_positions.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvato: {out}")


# ─────────────────────────────────────────────
# 5. NORMALIZZAZIONE E SALVATAGGIO
# ─────────────────────────────────────────────
def normalize_and_save(df: pd.DataFrame):
    """
    Fitta uno StandardScaler SOLO sulle feature di input (non sui target).
    Salva scaler e lista feature per riuso in training e inferenza.
    """
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    scaler.fit(X)

    # Salva scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler salvato: {scaler_path}")

    # Salva lista feature
    feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    with open(feature_path, "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    print(f"  Feature names salvate: {feature_path}")

    # Riporta statistiche scaler
    print("\n── Medie e std delle feature (per verifica) ──────────")
    for name, mean, std in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
        print(f"  {name:<14}: mean={mean:>8.3f}  std={std:>8.3f}")

    return scaler


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 1 – Preparazione Dataset")
    print("=" * 55)

    # 1. Carica
    print("\n[1/5] Caricamento CSV da dataset_laps/...")
    df_raw = load_all_laps(DATASET_DIR)

    # Salva merged grezzo
    merged_path = os.path.join(MODELS_DIR, "dataset_merged.csv")
    df_raw.to_csv(merged_path, index=False)
    print(f"  Dataset grezzo salvato: {merged_path}")

    # 2. Pulisci
    print("\n[2/5] Pulizia dati...")
    df = clean_data(df_raw)

    clean_path = os.path.join(MODELS_DIR, "dataset_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"  Dataset pulito salvato: {clean_path}")

    # 3. Statistiche
    print("\n[3/5] Statistiche descrittive...")
    print_stats(df)

    # 4. Plot
    print("\n[4/5] Generazione grafici EDA...")
    plot_distributions(df)
    plot_correlations(df)
    plot_track_positions(df)

    # 5. Normalizzazione
    print("\n[5/5] Normalizzazione e salvataggio scaler...")
    normalize_and_save(df)

    print("\n" + "=" * 55)
    print(f"  ✓ STEP 1 COMPLETATO")
    print(f"  Dataset pronto: {len(df)} campioni, {len(FEATURE_COLS)} feature")
    print(f"  Prossimo: python step2_train_knn.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
