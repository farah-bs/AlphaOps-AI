"""
Rolling Backtest Prophet — AlphaOps AI
=======================================
Walk-forward validation pour mesurer la qualité réelle des modèles Prophet.

Logique (aucune fuite de données garantie) :
    Pour chaque date cible >= start_date :
        1. Entraîner un Prophet FRAIS sur les `train_window` jours précédents
        2. Prédire uniquement le jour suivant (J+1, freq="B")
        3. Comparer yhat / [yhat_lower, yhat_upper] au prix réel

Métriques calculées :
    MAE, RMSE, MAPE, directional accuracy, interval coverage

Note sur les performances :
    Chaque step entraîne un nouveau modèle (~1-5s).
    Avec step=1, compter ~5-20 min par ticker.
    Utiliser step=5 (hebdo) ou step=21 (mensuel) pour accélérer.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from ml.features.feature_engineering import fetch_ohlcv
from ml.training.config import TrainingConfig

# Silencer les logs verbeux de Prophet/cmdstanpy
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Chemin vers artifacts/ à la racine du projet
# ml/evaluation/rolling_backtest.py → parent×3 = racine
ROOT      = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _ohlcv_to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un DataFrame OHLCV (DatetimeIndex) en format Prophet (ds, y).

    Priorité : adj_close → close_price (si adj_close contient des NaN).
    Supprime les valeurs infinies et NaN résiduelles.
    """
    prices = df["adj_close"].copy()

    # Fallback si adj_close est manquant pour certaines lignes
    if "close_price" in df.columns:
        prices = prices.fillna(df["close_price"])

    # Éliminer les valeurs aberrantes (inf, NaN)
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()

    return pd.DataFrame({
        "ds": pd.to_datetime(prices.index),
        "y":  prices.values,
    }).reset_index(drop=True)


def _fit_prophet_model(train_df: pd.DataFrame, cfg: TrainingConfig) -> Prophet:
    """
    Instancie et entraîne un Prophet sur train_df (format ds, y)
    avec les hyperparamètres définis dans TrainingConfig.
    """
    m = Prophet(
        n_changepoints=cfg.n_changepoints,
        uncertainty_samples=cfg.uncertainty_samples,
        interval_width=cfg.interval_width,
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        seasonality_mode=cfg.seasonality_mode,
        daily_seasonality=cfg.daily_seasonality,
        weekly_seasonality=cfg.weekly_seasonality,
        yearly_seasonality=cfg.yearly_seasonality,
    )
    m.fit(train_df)
    return m


# ── Backtest principal ─────────────────────────────────────────────────────────

def rolling_backtest_daily(
    ticker: str,
    train_window: int = 252,
    step: int = 1,
    start_date: str = "2024-01-01",
    cfg: Optional[TrainingConfig] = None,
    on_progress=None,  # callback(count, total, ticker, target_date) — appelé à chaque prédiction
) -> Tuple[pd.DataFrame, Dict]:
    """
    Rolling backtest journalier (walk-forward) sur un ticker Prophet.

    Pour chaque index pred_idx tel que ds[pred_idx] >= start_date :
      - Fenêtre d'entraînement = prophet_df[pred_idx - train_window : pred_idx]
      - Prédiction = J+1 après la fin de la fenêtre (make_future_dataframe periods=1, freq="B")
      - Comparaison à la valeur réelle prophet_df[pred_idx]

    Args:
        ticker:       Symbole (ex: "AAPL", "BTC-USD")
        train_window: Jours dans la fenêtre glissante (252 ≈ 1 an de trading)
        step:         Pas en nombre de lignes (1 = quotidien, 5 = hebdo, 21 = mensuel)
        start_date:   Première date cible à évaluer ("YYYY-MM-DD")
        cfg:          Configuration ; défaut = TrainingConfig()

    Returns:
        detail_df    : DataFrame avec une ligne par prédiction
        summary_dict : Métriques agrégées (MAE, RMSE, MAPE, dir. accuracy, coverage)
    """
    if cfg is None:
        cfg = TrainingConfig()

    # ── 1. Chargement des données depuis la DB ────────────────────────────────
    df_raw = fetch_ohlcv(ticker)
    if df_raw.empty:
        raise ValueError(f"Aucune donnée pour {ticker} dans fact_ohlcv.")

    prophet_df = _ohlcv_to_prophet_df(df_raw)
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    start_dt = pd.to_datetime(start_date)

    # ── 2. Trouver le premier indice de prédiction ────────────────────────────
    # Condition : avoir train_window lignes en amont ET être >= start_date
    first_pred_idx = None
    for i in range(train_window, len(prophet_df)):
        if prophet_df.loc[i, "ds"] >= start_dt:
            first_pred_idx = i
            break

    if first_pred_idx is None:
        raise ValueError(
            f"[{ticker}] Impossible de démarrer le backtest : "
            f"données insuffisantes ({len(prophet_df)} lignes) pour "
            f"train_window={train_window} avec start_date={start_date}."
        )

    # ── 3. Walk-forward loop ──────────────────────────────────────────────────
    records    = []
    pred_range = range(first_pred_idx, len(prophet_df), step)
    total      = len(pred_range)

    print(
        f"  [{ticker}] {total} prédictions "
        f"(train_window={train_window}j, step={step}j, "
        f"début={prophet_df.loc[first_pred_idx, 'ds'].date()})"
    )

    for count, pred_idx in enumerate(pred_range):
        # Fenêtre d'entraînement : train_window jours avant pred_idx (exclus)
        train_slice  = prophet_df.iloc[pred_idx - train_window : pred_idx].copy()

        # Prix cible (vrai) à la date pred_idx
        target_row   = prophet_df.iloc[pred_idx]
        target_date  = target_row["ds"]
        actual        = float(target_row["y"])

        # Dernier prix connu → sert au calcul de direction
        last_train_y = float(train_slice["y"].iloc[-1])

        # Entraîner + prédire J+1
        try:
            model    = _fit_prophet_model(train_slice, cfg)
            # freq="B" = jours ouvrés ; periods=1 = un seul jour après la dernière date connue
            future   = model.make_future_dataframe(periods=1, freq="B")
            forecast = model.predict(future)

            # La prédiction J+1 est toujours la dernière ligne du forecast
            last_row   = forecast.iloc[-1]
            yhat       = float(last_row["yhat"])
            yhat_lower = float(last_row["yhat_lower"])
            yhat_upper = float(last_row["yhat_upper"])

        except Exception as e:
            print(f"    SKIP idx={pred_idx} ({target_date.date()}) — {e}")
            continue

        # ── Métriques individuelles ───────────────────────────────────────────
        abs_error = abs(actual - yhat)
        pct_error = abs_error / (abs(actual) + 1e-8) * 100  # MAPE contribution

        # Direction : 1 = hausse, 0 = baisse, par rapport au dernier prix connu
        pred_dir          = int(yhat   > last_train_y)
        actual_dir        = int(actual > last_train_y)
        direction_correct = int(pred_dir == actual_dir)

        # Couverture : le vrai prix est-il dans l'intervalle [lower, upper] ?
        interval_hit = int(yhat_lower <= actual <= yhat_upper)

        records.append({
            "ticker":            ticker,
            "train_end":         train_slice["ds"].iloc[-1].date(),
            "target_date":       target_date.date(),
            "last_train_y":      last_train_y,
            "yhat":              yhat,
            "yhat_lower":        yhat_lower,
            "yhat_upper":        yhat_upper,
            "actual":            actual,
            "abs_error":         abs_error,
            "pct_error":         pct_error,
            "pred_dir":          pred_dir,
            "actual_dir":        actual_dir,
            "direction_correct": direction_correct,
            "interval_hit":      interval_hit,
        })

        # Callback UI (Streamlit) + affichage console toutes les 20 itérations
        if on_progress:
            on_progress(count + 1, total, ticker, target_date.date())
        elif (count + 1) % 20 == 0:
            print(f"    {count + 1}/{total} prédictions effectuées...")

    if not records:
        raise ValueError(
            f"[{ticker}] Aucune prédiction générée. "
            "Vérifiez que les données couvrent la période demandée."
        )

    # ── 4. Métriques globales ─────────────────────────────────────────────────
    detail_df = pd.DataFrame(records)

    actuals = detail_df["actual"].values
    yhats   = detail_df["yhat"].values

    mae      = float(np.mean(detail_df["abs_error"]))
    rmse     = float(np.sqrt(np.mean((actuals - yhats) ** 2)))
    mape     = float(np.mean(detail_df["pct_error"]))
    dir_acc  = float(detail_df["direction_correct"].mean()) * 100
    coverage = float(detail_df["interval_hit"].mean()) * 100

    summary = {
        "ticker":                   ticker,
        "start_date":               start_date,
        "train_window":             train_window,
        "step":                     step,
        "n_predictions":            len(detail_df),
        "mae":                      round(mae, 4),
        "rmse":                     round(rmse, 4),
        "mape_pct":                 round(mape, 4),
        "directional_accuracy_pct": round(dir_acc, 2),
        "interval_coverage_pct":    round(coverage, 2),
    }

    print(f"\n  [{ticker}] Résumé backtest :")
    print(f"    MAE   : {mae:.4f}")
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    MAPE  : {mape:.2f}%")
    print(f"    Dir.  : {dir_acc:.1f}%")
    print(f"    Cov.  : {coverage:.1f}%")

    return detail_df, summary


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_backtest_results(
    ticker: str,
    detail_df: pd.DataFrame,
    summary: Dict,
) -> None:
    """
    Sauvegarde les résultats dans artifacts/ :
      - backtest_<ticker>.csv           : détail ligne par ligne
      - backtest_summary_<ticker>.json  : métriques globales
    """
    ARTIFACTS.mkdir(exist_ok=True)

    detail_path  = ARTIFACTS / f"backtest_{ticker}.csv"
    summary_path = ARTIFACTS / f"backtest_summary_{ticker}.json"

    detail_df.to_csv(detail_path, index=False)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Sauvegardé : {detail_path.name}")
    print(f"  Sauvegardé : {summary_path.name}")


# ── Logging MLflow ────────────────────────────────────────────────────────────

def log_backtest_to_mlflow(
    ticker: str,
    detail_df: pd.DataFrame,
    summary: Dict,
    cfg: Optional[TrainingConfig] = None,
    csv_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
) -> None:
    """
    Log un run MLflow complet pour un rolling backtest ticker.

    Loggue : paramètres (ticker, fenêtre, step, hyperparams Prophet),
    métriques (MAE, RMSE, MAPE, directional accuracy, coverage),
    artefacts (CSV détail, JSON résumé, graphique actual vs yhat).

    Args:
        ticker     : symbole évalué (ex: "AAPL")
        detail_df  : DataFrame ligne par ligne du backtest
        summary    : dict de métriques agrégées
        cfg        : TrainingConfig utilisé (pour les hyperparams Prophet)
        csv_path   : chemin vers le CSV de détail déjà sauvegardé
        json_path  : chemin vers le JSON résumé déjà sauvegardé
    """
    import mlflow
    from ml.registry.mlflow_utils import (
        EXPERIMENT_BACKTEST, BASE_TAGS,
        get_or_create_experiment, log_params_safe,
        log_metrics_safe, log_artifact_path,
        log_dict_as_artifact, make_run_name, setup_mlflow,
    )

    if cfg is None:
        cfg = TrainingConfig()

    setup_mlflow()
    exp_id   = get_or_create_experiment(EXPERIMENT_BACKTEST)
    run_name = make_run_name("backtest", ticker)

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):

        # Tags
        mlflow.set_tags({
            **BASE_TAGS,
            "stage":     "evaluation",
            "eval_type": "rolling_backtest",
            "ticker":    ticker,
        })

        # Paramètres du backtest + hyperparams Prophet
        log_params_safe({
            "ticker":                   ticker,
            "train_window":             summary.get("train_window"),
            "step":                     summary.get("step"),
            "start_date":               summary.get("start_date"),
            "changepoint_prior_scale":  cfg.changepoint_prior_scale,
            "seasonality_prior_scale":  cfg.seasonality_prior_scale,
            "seasonality_mode":         cfg.seasonality_mode,
            "daily_seasonality":        cfg.daily_seasonality,
            "weekly_seasonality":       cfg.weekly_seasonality,
            "yearly_seasonality":       cfg.yearly_seasonality,
        })

        # Métriques
        log_metrics_safe({
            "mae":                      summary.get("mae"),
            "rmse":                     summary.get("rmse"),
            "mape_pct":                 summary.get("mape_pct"),
            "directional_accuracy_pct": summary.get("directional_accuracy_pct"),
            "interval_coverage_pct":    summary.get("interval_coverage_pct"),
            "n_predictions":            summary.get("n_predictions"),
        })

        # Artifacts : CSV + JSON déjà sauvegardés sur disque
        if csv_path:
            log_artifact_path(csv_path)
        if json_path:
            log_artifact_path(json_path)

        # Artifact : résumé JSON dans MLflow (sous-dossier summaries/)
        log_dict_as_artifact(summary, f"summary_{ticker}")

        # Artifact : graphique actual vs yhat (PNG)
        _log_backtest_figure(detail_df, ticker)

    print(f"  [MLflow] Run backtest {ticker} loggué.")


def _log_backtest_figure(detail_df: pd.DataFrame, ticker: str) -> None:
    """
    Génère et logue un graphique actual vs yhat en PNG dans MLflow.

    Utilise matplotlib (dépendance de Prophet, toujours disponible).
    Silencieux en cas d'erreur pour ne pas bloquer le backtest.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend non-interactif, compatible serveur
        import matplotlib.pyplot as plt
        import mlflow

        fig, ax = plt.subplots(figsize=(12, 5))

        # Intervalle de confiance
        ax.fill_between(
            detail_df["target_date"],
            detail_df["yhat_lower"],
            detail_df["yhat_upper"],
            alpha=0.18, color="#00d4ff", label="Intervalle [lower, upper]",
        )
        # Prix réel
        ax.plot(
            detail_df["target_date"], detail_df["actual"],
            color="#ff6b35", linewidth=1.5, label="Réel",
        )
        # Prédiction
        ax.plot(
            detail_df["target_date"], detail_df["yhat"],
            color="#00d4ff", linewidth=1.5, linestyle="--", label="yhat",
        )

        ax.set_title(f"{ticker} — Actual vs Yhat (Rolling Backtest)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=f"backtest_{ticker}_", delete=False,
        ) as f:
            tmp_path = Path(f.name)

        fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(tmp_path), artifact_path="figures")
        tmp_path.unlink(missing_ok=True)

    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[MLflow] Graphique non loggué pour {ticker} : {e}"
        )


# ── Multi-tickers ─────────────────────────────────────────────────────────────

def run_multi_ticker_backtest(
    tickers: Optional[List[str]] = None,
    train_window: int = 252,
    step: int = 1,
    start_date: str = "2024-01-01",
    cfg: Optional[TrainingConfig] = None,
) -> pd.DataFrame:
    """
    Lance le rolling backtest sur plusieurs tickers séquentiellement.

    Génère :
      - artifacts/backtest_<ticker>.csv        (détail par ticker)
      - artifacts/backtest_summary_<ticker>.json
      - artifacts/backtest_summary_all.csv     (synthèse globale)

    Args:
        tickers:      Liste de tickers (défaut = TrainingConfig().tickers)
        train_window: Fenêtre d'entraînement en jours
        step:         Pas du glissement
        start_date:   Première date cible
        cfg:          TrainingConfig

    Returns:
        DataFrame de synthèse (une ligne par ticker)
    """
    if cfg is None:
        cfg = TrainingConfig()
    if tickers is None:
        tickers = cfg.tickers

    all_summaries: List[Dict] = []

    for ticker in tickers:
        print(f"\n{'=' * 55}")
        print(f"  Backtest : {ticker}")
        print(f"{'=' * 55}")

        try:
            detail_df, summary = rolling_backtest_daily(
                ticker=ticker,
                train_window=train_window,
                step=step,
                start_date=start_date,
                cfg=cfg,
            )
            detail_path  = ARTIFACTS / f"backtest_{ticker}.csv"
            summary_path = ARTIFACTS / f"backtest_summary_{ticker}.json"
            save_backtest_results(ticker, detail_df, summary)

            # Logging MLflow (non-bloquant)
            try:
                log_backtest_to_mlflow(
                    ticker=ticker,
                    detail_df=detail_df,
                    summary=summary,
                    cfg=cfg,
                    csv_path=detail_path,
                    json_path=summary_path,
                )
            except Exception as mlflow_err:
                print(f"  [MLflow] Avertissement {ticker} : {mlflow_err}")

            all_summaries.append(summary)

        except Exception as e:
            print(f"  SKIP {ticker} — {e}")
            all_summaries.append({"ticker": ticker, "error": str(e)})

    # Résumé global multi-tickers
    summary_all_df   = pd.DataFrame(all_summaries)
    summary_all_path = ARTIFACTS / "backtest_summary_all.csv"
    summary_all_df.to_csv(summary_all_path, index=False)

    print(f"\nRésumé global sauvegardé : {summary_all_path.name}")
    return summary_all_df
