import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from prophet import Prophet
from ml.features.feature_engineering import fetch_ohlcv, compute_features
from ml.training.config import TrainingConfig

warnings.filterwarnings("ignore")

PROPHET_GRID = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.3, 0.5],
    "seasonality_prior_scale": [0.1, 1.0, 10.0],
    "seasonality_mode":        ["additive", "multiplicative"],
    "n_changepoints":          [15, 25, 40],
}

CFG       = TrainingConfig()
ROOT      = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _to_prophet_df(df: pd.DataFrame, regressors: list = None) -> pd.DataFrame:
    """
    Converts a full OHLCV DataFrame (DatetimeIndex) to Prophet format (ds, y).

    Uses the complete continuous time series — Prophet needs the full history
    to correctly estimate trend, changepoints, and seasonality components.
    Fragmented/windowed input would create artificial discontinuities and
    degrade forecast quality.

    If `regressors` is provided, adds those columns to the output df so that
    Prophet can use them as extra regressors during training.
    """
    out = df.copy()
    if "close_price" in out.columns:
        out["adj_close"] = out["adj_close"].fillna(out["close_price"])
    out["adj_close"] = out["adj_close"].replace([float("inf"), float("-inf")], float("nan"))
    out = out.dropna(subset=["adj_close"])

    result = pd.DataFrame({
        "ds": pd.to_datetime(out.index),
        "y":  out["adj_close"].values,
    }).reset_index(drop=True)

    if regressors:
        for reg in regressors:
            if reg in out.columns:
                result[reg] = out[reg].values

    return result


def _eval_prophet_direction(df_train: pd.DataFrame, df_val: pd.DataFrame, params: dict, regressors: list) -> float:
    """
    Entraîne Prophet avec `params` sur df_train, prédit sur df_val,
    retourne la direction accuracy (% de jours bien prédits).
    """
    try:
        m = Prophet(
            n_changepoints=params["n_changepoints"],
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            uncertainty_samples=0,  # rapide pour le grid search
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        for reg in regressors:
            if reg in df_train.columns:
                m.add_regressor(reg)
        m.fit(df_train)

        future = m.make_future_dataframe(periods=len(df_val), freq="B")
        for reg in regressors:
            if reg in df_train.columns:
                last_val = df_train[reg].iloc[-1]
                future[reg] = last_val
        forecast = m.predict(future)

        pred_vals = forecast.tail(len(df_val))["yhat"].values
        last_known = float(df_train["y"].iloc[-1])
        actual_dir = (df_val["y"].values > last_known).astype(int)
        pred_dir   = (pred_vals > last_known).astype(int)
        return float((pred_dir == actual_dir).mean())
    except Exception:
        return 0.0


def _prophet_grid_search(df: pd.DataFrame, cfg: TrainingConfig, val_days: int = 60) -> dict:
    """
    Grid search sur 90 combinaisons de hyperparamètres Prophet.
    Utilise les `val_days` derniers jours comme set de validation rapide.
    Retourne les meilleurs paramètres (dict).
    """
    split_idx = max(len(df) - val_days, int(len(df) * 0.8))
    df_train  = df.iloc[:split_idx]
    df_val    = df.iloc[split_idx:]

    if len(df_val) < 5:
        return {}  # pas assez de données pour évaluer

    keys   = list(PROPHET_GRID.keys())
    combos = list(itertools.product(*[PROPHET_GRID[k] for k in keys]))

    best_acc    = -1.0
    best_params = {}

    def _eval(combo):
        params = dict(zip(keys, combo))
        acc    = _eval_prophet_direction(df_train, df_val, params, cfg.prophet_regressors)
        return params, acc

    with ThreadPoolExecutor(max_workers=8) as ex:
        for params, acc in ex.map(_eval, combos):
            if acc > best_acc:
                best_acc    = acc
                best_params = params

    return best_params


def _fit_prophet(df: pd.DataFrame, cfg: TrainingConfig = None) -> Prophet:
    """
    Instancie et entraîne un modèle Prophet sur le DataFrame fourni.

    Paramètres clés pour la performance :
        - n_changepoints=25 (défaut) → détecte plus de ruptures de tendance
        - uncertainty_samples=1000 (défaut) → intervalles mieux estimés
    Pour la précision :
        - changepoint_prior_scale=0.1 → plus réactif aux changements de direction
        - seasonality_prior_scale=1.0 (vs 10.0) → moins de sur-ajustement
        - seasonality_mode=multiplicative → variation en % stable pour les actifs
        - extra_regressors : RSI, MACD, volatilité, log_return → signal directionnel
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
    for reg in c.prophet_regressors:
        if reg in df.columns:
            m.add_regressor(reg)
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

    df_d = _to_prophet_df(df_train, regressors=cfg.prophet_regressors)

    # ── Grid search hyperparamètres par ticker ──────────────────────────────
    print(f"  [{ticker}] grid search (90 combos)...")
    best_params = _prophet_grid_search(df_d, cfg, val_days=60)
    if best_params:
        import dataclasses
        cfg = dataclasses.replace(
            cfg,
            changepoint_prior_scale = best_params.get("changepoint_prior_scale", cfg.changepoint_prior_scale),
            seasonality_prior_scale = best_params.get("seasonality_prior_scale", cfg.seasonality_prior_scale),
            seasonality_mode        = best_params.get("seasonality_mode",        cfg.seasonality_mode),
            n_changepoints          = best_params.get("n_changepoints",          cfg.n_changepoints),
        )
        print(f"  [{ticker}] best params: {best_params}")

    # ── Daily model ────────────────────────────────────────────────────────
    print(f"  [{ticker}] daily   : {len(df_d):,} lignes (série continue)")
    model_d = _fit_prophet(df_d, cfg)

    # ── Monthly model ──────────────────────────────────────────────────────
    df_m = df_d.copy()
    print(f"  [{ticker}] monthly : {len(df_m):,} lignes (série continue)")
    model_m = _fit_prophet(df_m, cfg)

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
