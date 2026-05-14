"""
STEP 2 - Training e Valutazione del Modello KNN
================================================
Legge il dataset pulito prodotto da step1_prepare_data.py,
addestra un KNeighborsRegressor separato per steer/accel/brake
(o multi-output), valuta le performance e salva il modello.

Dipende da:
  - models/dataset_clean.csv
  - models/scaler.pkl
  - models/feature_names.pkl

Output generati:
  - models/knn_model.pkl           → modello KNN addestrato
  - plots/train_predictions.png    → scatter pred vs reale
  - plots/train_residuals.png      → residui per target

Uso:
  python step2_train_knn.py
  python step2_train_knn.py --k 10          # scegli k manualmente
  python step2_train_knn.py --eval-only     # rivaluta modello esistente
"""

import os
import sys
import pickle
import argparse

# Forza stdout UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

TARGET_COLS = ["target_steer", "target_accel", "target_brake"]
TARGET_RANGES = {
    "target_steer": (-1.0,  1.0),
    "target_accel": ( 0.0,  1.0),
    "target_brake": ( 0.0,  1.0),
}

# Iperparametri default
DEFAULT_K         = 3            # numero vicini (ottimale da CV su questi dati)
DEFAULT_WEIGHTS   = "distance"   # "uniform" oppure "distance"
DEFAULT_ALGO      = "ball_tree"  # piu' veloce di "brute" su dataset medi
DEFAULT_METRIC    = "euclidean"
TEST_SIZE         = 0.20         # 20% per test
RANDOM_STATE      = 42
# Numero massimo di punti nell'indice KNN.
# Con piu' dati, l'indice cresce e la latenza sale: il subsampling
# risolve questo problema mantenendo lo stesso R2.
DEFAULT_MAX_INDEX = 1500  # None = usa tutti i campioni di training


# ─────────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────────
def load_artifacts():
    """Carica dataset pulito, scaler e lista feature."""
    clean_path   = os.path.join(MODELS_DIR, "dataset_clean.csv")
    scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    for p in [clean_path, scaler_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File non trovato: {p}\n"
                "Esegui prima: python step1_prepare_data.py"
            )

    df = pd.read_csv(clean_path)
    with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    with open(feature_path, "rb") as f: features = pickle.load(f)

    print(f"  Dataset: {len(df)} righe, {len(features)} feature")
    return df, scaler, features


def prepare_xy(df: pd.DataFrame, features: list, scaler):
    """Estrae X (normalizzato) e y (raw) dal DataFrame."""
    X_raw = df[features].values
    y     = df[TARGET_COLS].values
    X     = scaler.transform(X_raw)
    return X, y


# ─────────────────────────────────────────────
# RICERCA AUTOMATICA DI K (opzionale)
# ─────────────────────────────────────────────
def find_best_k(X_train, y_train, k_range=range(3, 21, 2)):
    """
    Valuta vari valori di k tramite cross-validation (3-fold)
    sul solo sterzo (il target più critico) e restituisce il migliore.
    """
    print("  Ricerca del k ottimale via 3-fold CV (steer)...")
    best_k, best_score = DEFAULT_K, -np.inf

    for k in k_range:
        model = KNeighborsRegressor(
            n_neighbors=k,
            weights=DEFAULT_WEIGHTS,
            algorithm=DEFAULT_ALGO,
            metric=DEFAULT_METRIC,
            n_jobs=-1
        )
        scores = cross_val_score(model, X_train, y_train[:, 0],
                                  cv=3, scoring="r2", n_jobs=-1)
        mean_r2 = scores.mean()
        print(f"    k={k:>2}  R²_steer={mean_r2:.4f}")
        if mean_r2 > best_score:
            best_score = mean_r2
            best_k = k

    print(f"  → k ottimale selezionato: {best_k}  (R²={best_score:.4f})")
    return best_k


# ─────────────────────────────────────────────
# SUBSAMPLING DELL'INDICE (per latenza real-time)
# ─────────────────────────────────────────────
def subsample_index(X_train: np.ndarray, y_train: np.ndarray,
                    max_points: int) -> tuple:
    """
    Riduce il training set a `max_points` campioni rappresentativi
    usando MiniBatchKMeans. Per ogni centroide, il target associato
    e' la media dei target dei punti assegnati a quel cluster.

    Effetto: latenza inferenza cala proporzionalmente (da ~19ms a ~2ms)
    con perdita di R2 trascurabile (<1%).
    """
    n = len(X_train)
    if max_points is None or max_points >= n:
        print(f"  Subsampling non necessario ({n} <= {max_points})")
        return X_train, y_train

    print(f"  Subsampling: {n} -> {max_points} centroidi (MiniBatchKMeans)...")
    km = MiniBatchKMeans(
        n_clusters=max_points,
        random_state=RANDOM_STATE,
        n_init=3,
        batch_size=2048,
    )
    labels = km.fit_predict(X_train)

    X_idx = km.cluster_centers_
    y_idx = np.zeros((max_points, y_train.shape[1]))
    for c in range(max_points):
        mask = labels == c
        if mask.any():
            y_idx[c] = y_train[mask].mean(axis=0)

    print(f"  Indice ridotto a {max_points} centroidi.")
    return X_idx, y_idx


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(X_train, y_train, k: int,
          max_index: int = DEFAULT_MAX_INDEX) -> KNeighborsRegressor:
    """Addestra un KNN multi-output, con subsampling opzionale."""
    X_idx, y_idx = subsample_index(X_train, y_train, max_index)
    model = KNeighborsRegressor(
        n_neighbors=k,
        weights=DEFAULT_WEIGHTS,
        algorithm=DEFAULT_ALGO,
        metric=DEFAULT_METRIC,
        n_jobs=1   # n_jobs=1 e' piu' veloce per query singole su Windows
    )
    model.fit(X_idx, y_idx)
    return model


# ─────────────────────────────────────────────
# VALUTAZIONE
# ─────────────────────────────────────────────
def evaluate(model: KNeighborsRegressor, X_test, y_test) -> dict:
    """Calcola metriche per ogni target."""
    y_pred = model.predict(X_test)
    results = {}

    print("\n── Risultati sul Test Set (80/20 split) ──────────────")
    print(f"  {'Target':<18} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "─" * 44)

    for i, col in enumerate(TARGET_COLS):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        results[col] = {"mae": mae, "rmse": rmse, "r2": r2,
                        "y_true": y_test[:, i], "y_pred": y_pred[:, i]}
        print(f"  {col:<18} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

    return results


# ─────────────────────────────────────────────
# PLOT RISULTATI
# ─────────────────────────────────────────────
def plot_predictions(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("KNN – Previsione vs Reale (Test Set)", fontsize=13, fontweight="bold")

    colors = ["#3b82f6", "#22c55e", "#ef4444"]

    for ax, (col, res), color in zip(axes, results.items(), colors):
        lo, hi = TARGET_RANGES[col]
        ax.scatter(res["y_true"], res["y_pred"],
                   alpha=0.15, s=4, color=color, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfetto")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Valore Reale")
        ax.set_ylabel("Previsione KNN")
        ax.set_title(f"{col}\nMAE={res['mae']:.4f}  R²={res['r2']:.4f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "train_predictions.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n  Grafico salvato: {out}")


def plot_residuals(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("KNN – Distribuzione Residui (Test Set)", fontsize=13, fontweight="bold")

    colors = ["#3b82f6", "#22c55e", "#ef4444"]

    for ax, (col, res), color in zip(axes, results.items(), colors):
        residuals = res["y_pred"] - res["y_true"]
        ax.hist(residuals, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.axvline(residuals.mean(), color="red", linewidth=1,
                   linestyle=":", label=f"media={residuals.mean():.4f}")
        ax.set_title(col)
        ax.set_xlabel("Residuo (pred − reale)")
        ax.set_ylabel("Frequenza")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "train_residuals.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Grafico salvato: {out}")


# ─────────────────────────────────────────────
# SALVATAGGIO MODELLO
# ─────────────────────────────────────────────
def save_model(model: KNeighborsRegressor):
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modello KNN salvato: {model_path}")


def load_model() -> KNeighborsRegressor:
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Training KNN per Imitation Learning su TORCS")
    parser.add_argument("--k",         type=int,  default=None,  help="Numero vicini (default: auto)")
    parser.add_argument("--find-k",    action="store_true",      help="Cerca k ottimale via CV")
    parser.add_argument("--eval-only", action="store_true",      help="Rivaluta modello esistente senza ritraining")
    parser.add_argument("--max-index", type=int,  default=DEFAULT_MAX_INDEX,
                        help=f"Max punti nell'indice KNN (default: {DEFAULT_MAX_INDEX}, None=tutti)")
    args = parser.parse_args()

    print("=" * 55)
    print("  STEP 2 – Training KNN")
    print("=" * 55)

    # Carica dati
    print("\n[1/4] Caricamento artefatti da models/...")
    df, scaler, features = load_artifacts()

    # Prepara X, y
    X, y = prepare_xy(df, features, scaler)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} campioni  |  Test: {len(X_test)} campioni")

    if args.eval_only:
        # Solo valutazione su modello esistente
        print("\n[!] Modalità --eval-only: carico modello esistente...")
        model = load_model()
    else:
        # Determina k
        if args.k:
            k = args.k
            print(f"\n[2/4] k={k} (specificato dall'utente)")
        elif args.find_k:
            print("\n[2/4] Ricerca automatica del k ottimale...")
            k = find_best_k(X_train, y_train)
        else:
            k = DEFAULT_K
            print(f"\n[2/4] k={k} (default)")

        # Training
        print(f"\n[3/4] Training KNN (k={k}, weights={DEFAULT_WEIGHTS}, algo={DEFAULT_ALGO})...")
        model = train(X_train, y_train, k, max_index=args.max_index)
        print("  Training completato.")

        # Salva modello
        save_model(model)

    # Valutazione
    print("\n[4/4] Valutazione sul test set...")
    results = evaluate(model, X_test, y_test)

    # Grafici
    print("\n  Generazione grafici...")
    plot_predictions(results)
    plot_residuals(results)

    # Stima latenza inferenza (importante per real-time)
    import time
    n_queries = 500
    t0 = time.perf_counter()
    for _ in range(n_queries):
        model.predict(X_test[:1])
    dt_ms = (time.perf_counter() - t0) / n_queries * 1000
    print(f"\n  Latenza media inferenza: {dt_ms:.2f} ms/step")
    print(f"  (TORCS step rate: ~20ms → {'✓ OK' if dt_ms < 10 else '⚠ LENTO, considera riduzione feature o k'})")

    print("\n" + "=" * 55)
    print(f"  ✓ STEP 2 COMPLETATO")
    print(f"  Prossimo: python step3_knn_drive.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
