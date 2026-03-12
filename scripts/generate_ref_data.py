"""
Génère data/ref_data.csv et artifacts/scaler.pickle.
À lancer une fois depuis la racine du projet :
    python scripts/generate_ref_data.py
"""
import sys
import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features.feature_engineering import fetch_ohlcv, compute_features, FEATURE_COLS
from ml.training.config import TrainingConfig

CFG       = TrainingConfig()
ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA      = ROOT / "data"
ARTIFACTS.mkdir(exist_ok=True)


def build_windows(df: pd.DataFrame, window: int) -> list:
    """
    Fenêtres non-overlapping de `window` jours.
    Chaque sample = features du dernier jour de la fenêtre
                  + target = (close[J+1] > close[J])
    """
    rows = []
    prices   = df["adj_close"].values
    features = df[FEATURE_COLS].values
    dates    = df.index.values

    # pas = window → zéro overlap entre fenêtres
    for start in range(0, len(df) - window, window):
        end = start + window
        if end >= len(df):
            break
        rows.append({
            "date":   pd.Timestamp(dates[end - 1]),
            **{f: v for f, v in zip(FEATURE_COLS, features[end - 1])},
            "target": bool(prices[end] > prices[end - 1]),
        })
    return rows


def main():
    all_rows = []

    for ticker in CFG.tickers:
        print(f"  {ticker}...")
        try:
            df = fetch_ohlcv(ticker)
            df = compute_features(df)
        except Exception as e:
            print(f"    SKIP — {e}")
            continue

        # split train uniquement → pas de data leakage
        df_train = df[df.index <= CFG.train_end]
        rows = build_windows(df_train, CFG.daily_window)
        for r in rows:
            r["symbol"] = ticker
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("Aucune donnée collectée — la DB est-elle démarrée ?")

    ref = pd.DataFrame(all_rows)

    # Scaler fitté uniquement sur les features du split train
    scaler = StandardScaler()
    ref[FEATURE_COLS] = scaler.fit_transform(ref[FEATURE_COLS].values)

    # Ordre des colonnes conforme au sujet
    ref = ref[["symbol", "date"] + FEATURE_COLS + ["target"]]
    ref.to_csv(DATA / "ref_data.csv", index=False)
    print(f"\nref_data.csv sauvegardé  ({len(ref)} lignes, {len(CFG.tickers)} tickers)")

    with open(ARTIFACTS / "scaler.pickle", "wb") as f:
        pickle.dump(scaler, f)
    print("scaler.pickle sauvegardé")


if __name__ == "__main__":
    main()
