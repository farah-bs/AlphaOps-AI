#!/usr/bin/env python3
"""
Script CLI — Rolling Backtest Prophet (AlphaOps AI)
====================================================
Lance le walk-forward validation sur un ou plusieurs tickers,
sauvegarde les artifacts dans artifacts/, et affiche les métriques.

Exemples d'usage :
    # Backtest sur un ticker
    python scripts/run_backtest.py --ticker AAPL

    # Backtest avec options
    python scripts/run_backtest.py --ticker AAPL --start-date 2024-06-01 --step 5

    # Backtest sur tous les tickers de TrainingConfig
    python scripts/run_backtest.py

    # Version plus rapide (pas hebdomadaire, ~5x plus rapide)
    python scripts/run_backtest.py --step 5

    # Fenêtre d'entraînement plus courte (6 mois)
    python scripts/run_backtest.py --ticker MSFT --train-window 126

Note sur les performances :
    step=1  → prédiction quotidienne, très précis, ~5-20 min/ticker
    step=5  → prédiction hebdo, ~1-4 min/ticker
    step=21 → prédiction mensuelle, très rapide, moins de détails
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ajoute la racine du projet au PYTHONPATH pour permettre les imports ml.*, src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.evaluation.rolling_backtest import (
    log_backtest_to_mlflow,
    rolling_backtest_daily,
    run_multi_ticker_backtest,
    save_backtest_results,
)
from ml.training.config import TrainingConfig


def _print_summary(summary: dict) -> None:
    """Affiche un résumé formaté des métriques pour un ticker."""
    print(f"\n{'─' * 45}")
    print(f"  Résumé — {summary.get('ticker', '?')}")
    print(f"{'─' * 45}")

    labels = {
        "start_date":               "Début du backtest",
        "train_window":             "Fenêtre d'entraînement (j)",
        "step":                     "Pas (j)",
        "n_predictions":            "Nb prédictions",
        "mae":                      "MAE ($)",
        "rmse":                     "RMSE ($)",
        "mape_pct":                 "MAPE (%)",
        "directional_accuracy_pct": "Directional Accuracy (%)",
        "interval_coverage_pct":    "Interval Coverage (%)",
    }

    for key, label in labels.items():
        if key in summary:
            print(f"  {label:<34}: {summary[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling Backtest Prophet — AlphaOps AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker à évaluer (ex: AAPL). Absent = tous les tickers.",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=252,
        help="Jours dans la fenêtre glissante (défaut: 252 ≈ 1 an de trading).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Pas du glissement en jours (défaut: 1). step=5 est ~5x plus rapide.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Première date cible de prédiction (défaut: 2024-01-01).",
    )

    args = parser.parse_args()
    cfg  = TrainingConfig()

    print("=" * 60)
    print("  AlphaOps AI — Rolling Backtest Prophet")
    print("=" * 60)
    print(f"  start_date   : {args.start_date}")
    print(f"  train_window : {args.train_window} jours")
    print(f"  step         : {args.step} jour(s)")

    if args.ticker:
        # ── Mode un seul ticker ───────────────────────────────────────────────
        ticker = args.ticker.upper()
        print(f"  ticker       : {ticker}")
        print()

        try:
            detail_df, summary = rolling_backtest_daily(
                ticker=ticker,
                train_window=args.train_window,
                step=args.step,
                start_date=args.start_date,
                cfg=cfg,
            )
            detail_path  = ROOT / "artifacts" / f"backtest_{ticker}.csv"
            summary_path = ROOT / "artifacts" / f"backtest_summary_{ticker}.json"
            save_backtest_results(ticker, detail_df, summary)
            _print_summary(summary)

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
                print(f"\n[MLflow] Avertissement : {mlflow_err}")

        except Exception as e:
            print(f"\nERREUR : {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # ── Mode multi-tickers ────────────────────────────────────────────────
        print(f"  tickers      : {', '.join(cfg.tickers)}")
        print()

        summary_df = run_multi_ticker_backtest(
            tickers=cfg.tickers,
            train_window=args.train_window,
            step=args.step,
            start_date=args.start_date,
            cfg=cfg,
        )

        print(f"\n{'=' * 60}")
        print("  Résumé global")
        print(f"{'=' * 60}")

        # Séparer tickers réussis et tickers en erreur
        if "error" in summary_df.columns:
            ok  = summary_df[summary_df["error"].isna()].drop(columns=["error"], errors="ignore")
            err = summary_df[summary_df["error"].notna()]
        else:
            ok  = summary_df
            err = pd.DataFrame()

        if not ok.empty:
            metric_cols = [c for c in [
                "ticker", "n_predictions", "mae", "rmse",
                "mape_pct", "directional_accuracy_pct", "interval_coverage_pct",
            ] if c in ok.columns]
            print(ok[metric_cols].to_string(index=False))

        if not err.empty:
            print(f"\nTickers en erreur : {err['ticker'].tolist()}")


if __name__ == "__main__":
    main()
