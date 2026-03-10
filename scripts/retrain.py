import pickle
import warnings
import pandas as pd
from pathlib import Path
from ml.features.feature_engineering import fetch_ohlcv, compute_features
from ml.training.config import TrainingConfig
from ml.training.train import _build_training_df, _fit_prophet

warnings.filterwarnings("ignore")

CFG       = TrainingConfig()
ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA      = ROOT / "data"


def retrain(cfg: TrainingConfig = CFG) -> dict:
    """
    Réentraîne les modèles daily et monthly sur toutes les données disponibles.
    Re-fetch les prix bruts depuis la DB pour chaque ticker (données à jour).
    Met à jour les artifacts pickle.

    Retourne les nouveaux modèles pour mise à jour des variables globales de l'API.
    """
    # Compter les données prod disponibles pour le log
    prod_path = DATA / "prod_data.csv"
    n_prod    = 0
    if prod_path.exists() and prod_path.stat().st_size > 0:
        n_prod = len(pd.read_csv(prod_path))

    ref    = pd.read_csv(DATA / "ref_data.csv")
    n_ref  = len(ref)
    print(f"Réentraînement — {n_ref} ref + {n_prod} prod = {n_ref + n_prod} lignes total")

    models_daily:   dict = {}
    models_monthly: dict = {}

    for ticker in cfg.tickers:
        print(f"\n  [{ticker}]")
        try:
            df_raw = fetch_ohlcv(ticker)
            df_raw = compute_features(df_raw)
        except Exception as e:
            print(f"    SKIP — {e}")
            continue

        if len(df_raw) < cfg.monthly_window + 1:
            print(f"    SKIP — pas assez de lignes ({len(df_raw)})")
            continue

        # Daily (fenêtres 60j)
        df_d = _build_training_df(df_raw, cfg.daily_window)
        print(f"    daily   : {len(df_d):,} lignes")
        models_daily[ticker] = _fit_prophet(df_d)

        # Monthly (fenêtres 180j)
        df_m = _build_training_df(df_raw, cfg.monthly_window)
        print(f"    monthly : {len(df_m):,} lignes")
        models_monthly[ticker] = _fit_prophet(df_m)

    if not models_daily:
        raise RuntimeError("Aucun modèle réentraîné — la DB est-elle démarrée ?")

    with open(ARTIFACTS / "models_daily.pickle",   "wb") as f:
        pickle.dump(models_daily, f)
    with open(ARTIFACTS / "models_monthly.pickle", "wb") as f:
        pickle.dump(models_monthly, f)

    print("\nArtifacts mis à jour : models_daily.pickle, models_monthly.pickle")
    return {"models_daily": models_daily, "models_monthly": models_monthly}


if __name__ == "__main__":
    retrain()
