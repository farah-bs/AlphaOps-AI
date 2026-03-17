import pickle
import warnings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _fit_prophet(df: pd.DataFrame, cfg: TrainingConfig = None) -> Prophet:
    """
    Instancie et entraîne un modèle Prophet sur le DataFrame fourni.

    Paramètres clés pour la performance :
        - n_changepoints réduit (15 vs 25 défaut) → fit ~30% plus rapide
        - uncertainty_samples réduit (300 vs 1000 défaut) → predict 3× plus rapide
    Pour la précision :
        - seasonality_prior_scale=1.0 (vs 10.0) → moins de sur-ajustement
        - seasonality_mode=multiplicative → variation en % stable pour les actifs
    """
    c = cfg or CFG
    m = Prophet(
        n_changepoints=c.n_changepoints,
        uncertainty_samples=c.uncertainty_samples,
        interval_width=c.interval_width,
        changepoint_prior_scale=c.changepoint_prior_scale,
        seasonality_prior_scale=c.seasonality_prior_scale,
        seasonality_mode=c.seasonality_mode,
        daily_seasonality=c.daily_seasonality,
        weekly_seasonality=c.weekly_seasonality,
        yearly_seasonality=c.yearly_seasonality,
    )
    m.fit(df)
    return m


def _train_ticker(args: tuple) -> tuple:
    """
    Entraîne les modèles daily et monthly pour un ticker.
    Fonction au niveau module pour être utilisable dans ThreadPoolExecutor.

    Returns:
        (ticker, model_daily, model_monthly, mean_price, std_price)
    """
    ticker, df_train, cfg = args

    # ── Daily model ────────────────────────────────────────────────────────
    df_d       = _build_training_df(df_train, cfg.daily_window)
    n_d        = len(df_d) // cfg.daily_window
    print(f"  [{ticker}] daily   : {len(df_d):,} lignes — {n_d} fenêtres de {cfg.daily_window}j")
    model_d    = _fit_prophet(df_d, cfg)

    # ── Monthly model ──────────────────────────────────────────────────────
    df_m       = _build_training_df(df_train, cfg.monthly_window)
    n_m        = len(df_m) // cfg.monthly_window
    print(f"  [{ticker}] monthly : {len(df_m):,} lignes — {n_m} fenêtres de {cfg.monthly_window}j")
    model_m    = _fit_prophet(df_m, cfg)

    mean_ = float(df_train["adj_close"].mean())
    std_  = float(df_train["adj_close"].std())

    return ticker, model_d, model_m, mean_, std_


def train(cfg: TrainingConfig = CFG) -> dict:
    """
    Pipeline complet d'entraînement.

    Optimisations :
        - Récupération des données en séquentiel (I/O DB)
        - Entraînement Prophet en parallèle par ticker (ThreadPoolExecutor)
          → Prophet utilise CmdStan (code natif) qui libère le GIL
          → gain réel de parallélisme même avec le GIL Python

    Retourne un dict avec models_daily, models_monthly et price_stats.
    """
    print(f"\n  [Config] train_end = {cfg.train_end}")
    print(f"  [Config] uncertainty_samples={cfg.uncertainty_samples}, "
          f"n_changepoints={cfg.n_changepoints}, "
          f"seasonality_mode={cfg.seasonality_mode}\n")

    # ── 1. Récupération et préparation des données (séquentiel) ───────────────
    ticker_args = []
    for ticker in cfg.tickers:
        print(f"\n  [{ticker}] Chargement données...")
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

        ticker_args.append((ticker, df_train, cfg))

    if not ticker_args:
        raise RuntimeError("Aucun ticker disponible — la DB est-elle démarrée ?")

    # ── 2. Entraînement parallèle (4 tickers simultanément) ──────────────────
    # Prophet libère le GIL via CmdStan → ThreadPoolExecutor donne du vrai parallélisme
    print(f"\n  Entraînement parallèle de {len(ticker_args)} tickers (max 4 simultanés)...\n")

    models_daily:   dict = {}
    models_monthly: dict = {}
    price_stats:    dict = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_train_ticker, args): args[0]
            for args in ticker_args
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, m_d, m_m, mean_, std_ = future.result()
                models_daily[t]   = m_d
                models_monthly[t] = m_m
                price_stats[t]    = (mean_, std_)
                print(f"  [{t}] ✓ entraînement terminé")
            except Exception as e:
                print(f"  [{ticker}] ERREUR — {e}")

    if not models_daily:
        raise RuntimeError("Aucun modèle entraîné — vérifier les logs ci-dessus")

    # ── 3. Sauvegarde des artifacts ───────────────────────────────────────────
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

    # ── 4. Logging MLflow (params + métriques + pickles, sans log_model) ──────
    # Note : mlflow.prophet.log_model est désactivé ici car il sérialise chaque
    # modèle Prophet en JSON (~1-2 min/modèle × 18 = 18-36 min inutiles).
    # Les pickles sont loggués à la place pour la traçabilité.
    try:
        _log_training_to_mlflow(cfg, models_daily, models_monthly, price_stats)
    except Exception as e:
        print(f"  [MLflow] Avertissement : logging ignoré — {e}")

    return {
        "models_daily":   models_daily,
        "models_monthly": models_monthly,
        "price_stats":    price_stats,
    }


def _log_training_to_mlflow(
    cfg: TrainingConfig,
    models_daily: dict,
    models_monthly: dict,
    price_stats: dict,
) -> None:
    """
    Log un run MLflow : paramètres Prophet, métriques (nb modèles), artifacts (pickles).
    Le log_model Prophet JSON (lent) est exclu — les pickles suffisent pour la traçabilité.
    """
    import mlflow
    from ml.registry.mlflow_utils import (
        EXPERIMENT_TRAINING, BASE_TAGS,
        get_or_create_experiment, log_params_safe,
        log_metrics_safe, log_artifact_path,
        log_dict_as_artifact, make_run_name, setup_mlflow,
    )

    setup_mlflow()
    exp_id   = get_or_create_experiment(EXPERIMENT_TRAINING)
    run_name = make_run_name("train_prophet")

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags({**BASE_TAGS, "stage": "training"})

        log_params_safe({
            "tickers":                    cfg.tickers,
            "train_end":                  cfg.train_end,
            "val_end":                    cfg.val_end,
            "daily_window":               cfg.daily_window,
            "daily_horizon":              cfg.daily_horizon,
            "monthly_window":             cfg.monthly_window,
            "monthly_horizon":            cfg.monthly_horizon,
            "n_changepoints":             cfg.n_changepoints,
            "uncertainty_samples":        cfg.uncertainty_samples,
            "interval_width":             cfg.interval_width,
            "changepoint_prior_scale":    cfg.changepoint_prior_scale,
            "seasonality_prior_scale":    cfg.seasonality_prior_scale,
            "seasonality_mode":           cfg.seasonality_mode,
        })

        log_metrics_safe({
            "n_tickers_trained":  len(models_daily),
            "n_models_daily":     len(models_daily),
            "n_models_monthly":   len(models_monthly),
        })

        for fname in ["models_daily.pickle", "models_monthly.pickle", "price_stats.pickle"]:
            log_artifact_path(ARTIFACTS / fname)

        tickers_summary = {
            "tickers_trained": list(models_daily.keys()),
            "n_tickers":       len(models_daily),
            "train_end":       cfg.train_end,
            "price_stats":     {
                t: {"mean": round(v[0], 4), "std": round(v[1], 4)}
                for t, v in price_stats.items()
            },
        }
        log_dict_as_artifact(tickers_summary, "tickers_summary")

    print("  [MLflow] Run entraînement loggué.")


if __name__ == "__main__":
    print("=== Entraînement des modèles ===\n")
    train()
    print("\nTerminé.")
