import pickle
import warnings
import pandas as pd
from pathlib import Path
from prophet import Prophet
from ml.features.feature_engineering import fetch_ohlcv, compute_features
from ml.training.config import TrainingConfig

warnings.filterwarnings("ignore")

CFG       = TrainingConfig()
ROOT      = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _build_training_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Concatène les fenêtres non-overlapping de `window` jours en un seul
    DataFrame Prophet (ds, y) avec les vraies dates calendaires.

    step = window → zéro overlap garanti entre fenêtres.
    """
    frames = []
    close  = df["adj_close"].values
    dates  = df.index

    for start in range(0, len(df) - window, window):
        end = start + window
        frames.append(pd.DataFrame({
            "ds": dates[start:end],
            "y":  close[start:end],
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit_prophet(df: pd.DataFrame) -> Prophet:
    """Instancie et entraîne un modèle Prophet sur le DataFrame fourni."""
    m = Prophet(
        changepoint_prior_scale=CFG.changepoint_prior_scale,
        seasonality_prior_scale=CFG.seasonality_prior_scale,
        seasonality_mode=CFG.seasonality_mode,
        daily_seasonality=CFG.daily_seasonality,
        weekly_seasonality=CFG.weekly_seasonality,
        yearly_seasonality=CFG.yearly_seasonality,
    )
    m.fit(df)
    return m


def train(cfg: TrainingConfig = CFG) -> dict:
    """
    Pipeline complet d'entraînement.
    Retourne un dict avec models_daily, models_monthly et price_stats.
    """
    models_daily:   dict = {}
    models_monthly: dict = {}
    price_stats:    dict = {}

    for ticker in cfg.tickers:
        print(f"\n  [{ticker}]")
        try:
            df = fetch_ohlcv(ticker)
            df = compute_features(df)
        except Exception as e:
            print(f"    SKIP — {e}")
            continue

        df_train = df[df.index <= cfg.train_end].copy()

        if len(df_train) < cfg.monthly_window + 1:
            print(f"    SKIP — pas assez de lignes ({len(df_train)})")
            continue

        price_stats[ticker] = (
            float(df_train["adj_close"].mean()),
            float(df_train["adj_close"].std()),
        )

        # ── Daily model (fenêtres 60j) ────────────────────────────────────
        df_d = _build_training_df(df_train, cfg.daily_window)
        n_windows_d = len(df_d) // cfg.daily_window
        print(f"    daily   : {len(df_d):,} lignes — {n_windows_d} fenêtres de {cfg.daily_window}j")
        models_daily[ticker] = _fit_prophet(df_d)

        # ── Monthly model (fenêtres 180j) ─────────────────────────────────
        df_m = _build_training_df(df_train, cfg.monthly_window)
        n_windows_m = len(df_m) // cfg.monthly_window
        print(f"    monthly : {len(df_m):,} lignes — {n_windows_m} fenêtres de {cfg.monthly_window}j")
        models_monthly[ticker] = _fit_prophet(df_m)

    if not models_daily:
        raise RuntimeError("Aucun modèle entraîné — la DB est-elle démarrée ?")

    # ── Sauvegarde des artifacts ──────────────────────────────────────────────
    with open(ARTIFACTS / "models_daily.pickle",   "wb") as f:
        pickle.dump(models_daily, f)
    with open(ARTIFACTS / "models_monthly.pickle", "wb") as f:
        pickle.dump(models_monthly, f)
    with open(ARTIFACTS / "price_stats.pickle",    "wb") as f:
        pickle.dump(price_stats, f)

    print("\nArtifacts sauvegardés :")
    print("  artifacts/models_daily.pickle")
    print("  artifacts/models_monthly.pickle")
    print("  artifacts/price_stats.pickle")

    return {
        "models_daily":   models_daily,
        "models_monthly": models_monthly,
        "price_stats":    price_stats,
    }


if __name__ == "__main__":
    print("=== Entraînement des modèles ===\n")
    train()
    print("\nTerminé.")
