"""CLI — Monitoring Evidently
Usage :
    python scripts/run_monitoring.py                   # tous les tickers
    python scripts/run_monitoring.py --ticker AAPL     # un seul ticker
    python scripts/run_monitoring.py --ticker AAPL MSFT NVDA
"""

import argparse
import sys
from pathlib import Path

# Rend les modules du projet importables
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.monitoring.monitor import TICKERS_DEFAULT, run_monitoring


def main():
    parser = argparse.ArgumentParser(description="Lance le monitoring Evidently AlphaOps")
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=TICKERS_DEFAULT,
        help="Liste de tickers (défaut : tous)",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.ticker]
    print(f"\nMonitoring AlphaOps — {len(tickers)} ticker(s)")
    print("=" * 60)

    for ticker in tickers:
        print(f"\n{'─' * 60}")
        print(f"  Ticker : {ticker}")
        print(f"{'─' * 60}")
        try:
            summary = run_monitoring(ticker)
            _print_summary(summary)
        except Exception as exc:
            print(f"  ERREUR : {exc}")

    print("\n✅ Monitoring terminé.")
    print(f"   Rapports HTML → {ROOT / 'artifacts' / 'monitoring_reports'}/")
    print(f"   Résumés JSON  → {ROOT / 'artifacts'}/monitoring_summary_*.json")


def _print_summary(s: dict) -> None:
    """Affiche un résumé formaté dans le terminal."""
    print(f"  Run : {s.get('run_date', '—')}")

    q = s.get("data_quality", {})
    if "error" in q:
        print(f"  [Qualité]      ERREUR : {q['error']}")
    else:
        print(
            f"  [Qualité]      {q.get('missing_values_pct', '?')}% manquants"
            f" — {q.get('n_rows', '?')} lignes"
        )

    d = s.get("data_drift", {})
    if "error" in d:
        print(f"  [Drift]        ERREUR : {d['error']}")
    else:
        flag  = "🔴 DÉTECTÉ" if d.get("dataset_drift") else "🟢 OK"
        cols  = d.get("drifted_columns", [])
        extra = f" ({', '.join(cols)})" if cols else ""
        print(
            f"  [Drift]        {flag}"
            f" — {d.get('n_drifted', '?')}/{d.get('n_features', '?')} features{extra}"
        )

    p = s.get("model_performance", {})
    if "error" in p:
        print(f"  [Performance]  ERREUR : {p['error']}")
    else:
        deg  = p.get("mae_degradation_pct")
        sign = "+" if (deg or 0) > 0 else ""
        print(
            f"  [Performance]  MAE courant={p.get('mae_current', '?')}"
            f"  réf={p.get('mae_reference', '?')}"
            + (f"  Δ={sign}{deg:.1f}%" if deg is not None else "")
        )


if __name__ == "__main__":
    main()
