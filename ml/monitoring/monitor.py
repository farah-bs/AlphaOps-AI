"""Monitoring automatisé AlphaOps — Evidently
============================================
Trois volets :
  1. Qualité des données  (DataQualityPreset)   — features courantes
  2. Dérive des features  (DataDriftPreset)     — référence 2023 vs courant 2024+
  3. Performance modèle   (RegressionPreset)    — backtest 1ère moitié vs 2ème moitié

Sorties dans artifacts/ :
  monitoring_reports/quality_<TICKER>.html
  monitoring_reports/drift_<TICKER>.html
  monitoring_reports/performance_<TICKER>.html
  monitoring_summary_<TICKER>.json
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
from evidently.legacy.report import Report

# ── Chemins ───────────────────────────────────────────────────────────────────
# monitor.py → ml/monitoring/ → ml/ → racine projet
_ROOT     = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _ROOT / "artifacts"
REPORTS   = ARTIFACTS / "monitoring_reports"

# Import depuis feature_engineering pour rester synchronisé automatiquement
from ml.features.feature_engineering import FEATURE_COLS


def _get_monitoring_windows():
    """
    Returns dynamic monitoring windows based on today's date.

    Reference : last 12 months ending 90 days ago
                (represents recent training distribution)
    Current   : last 90 days
                (represents what the model sees in production)

    This ensures monitoring always compares a stable recent-training window
    against the live production window, without hardcoded dates that go stale.
    """
    today         = datetime.now().date()
    current_end   = today
    current_start = today - timedelta(days=90)
    ref_end       = current_start - timedelta(days=1)
    ref_start     = ref_end - timedelta(days=365)
    return (
        str(ref_start),
        str(ref_end),
        str(current_start),
        str(current_end),
    )

TICKERS_DEFAULT = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD",
]


# ── Point d'entrée principal ──────────────────────────────────────────────────

def run_monitoring(ticker: str) -> dict:
    """
    Lance le monitoring complet pour un ticker.

    Génère 3 rapports HTML + un JSON de synthèse dans artifacts/.
    Retourne le dict de synthèse.

    Args:
        ticker : ex. "AAPL"

    Returns:
        dict avec les clés ticker, run_date, data_quality, data_drift, model_performance
    """
    REPORTS.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "ticker":            ticker,
        "run_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_quality":      {},
        "data_drift":        {},
        "model_performance": {},
    }

    # ── 1. Qualité + Dérive des features ──────────────────────────────────────
    try:
        from ml.features.feature_engineering import compute_features, fetch_ohlcv

        ref_start, ref_end, curr_start, curr_end = _get_monitoring_windows()

        df      = fetch_ohlcv(ticker)
        df      = compute_features(df)
        df      = df.reset_index()   # DatetimeIndex → colonne "date"

        df_ref  = (
            df[(df["date"] >= ref_start) & (df["date"] <= ref_end)][FEATURE_COLS]
            .copy().reset_index(drop=True)
        )
        df_curr = (
            df[(df["date"] >= curr_start) & (df["date"] <= curr_end)][FEATURE_COLS]
            .copy().reset_index(drop=True)
        )

        summary["monitoring_windows"] = {
            "reference": f"{ref_start} → {ref_end}",
            "current":   f"{curr_start} → {curr_end}",
        }

        if len(df_ref) < 10 or len(df_curr) < 10:
            raise ValueError(
                f"Données insuffisantes — référence : {len(df_ref)} lignes, "
                f"courant : {len(df_curr)} lignes (minimum 10 requis)"
            )

        # Rapport qualité
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(reference_data=df_ref, current_data=df_curr)
        quality_report.save_html(str(REPORTS / f"quality_{ticker}.html"))
        summary["data_quality"] = _extract_quality(quality_report.as_dict())

        # Rapport drift
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=df_ref, current_data=df_curr)
        drift_report.save_html(str(REPORTS / f"drift_{ticker}.html"))
        summary["data_drift"] = _extract_drift(drift_report.as_dict())

    except Exception as exc:
        summary["data_quality"]["error"] = str(exc)
        summary["data_drift"]["error"]   = str(exc)

    # ── 2. Performance modèle (depuis le backtest CSV) ─────────────────────────
    backtest_path = ARTIFACTS / f"backtest_{ticker}.csv"
    if backtest_path.exists():
        try:
            df_bt = pd.read_csv(backtest_path, parse_dates=["target_date"])

            # Evidently RegressionPreset attend les colonnes "target" et "prediction"
            df_bt = df_bt.rename(columns={"actual": "target", "yhat": "prediction"})

            # Split temporel : 1ère moitié = référence, 2ème moitié = courant
            mid = len(df_bt) // 2
            df_ref_p  = df_bt.iloc[:mid][["target", "prediction"]].copy().reset_index(drop=True)
            df_curr_p = df_bt.iloc[mid:][["target", "prediction"]].copy().reset_index(drop=True)

            if len(df_ref_p) < 5 or len(df_curr_p) < 5:
                raise ValueError("Pas assez de prédictions dans le backtest (< 5)")

            perf_report = Report(metrics=[RegressionPreset()])
            perf_report.run(reference_data=df_ref_p, current_data=df_curr_p)
            perf_report.save_html(str(REPORTS / f"performance_{ticker}.html"))
            summary["model_performance"] = _extract_performance(perf_report.as_dict())

        except Exception as exc:
            summary["model_performance"]["error"] = str(exc)
    else:
        summary["model_performance"]["error"] = (
            "Backtest introuvable — lancez d'abord le backtest "
            "depuis l'onglet 📊 Backtest"
        )

    # ── Sauvegarde résumé JSON ─────────────────────────────────────────────────
    summary_path = ARTIFACTS / f"monitoring_summary_{ticker}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ── Helpers d'extraction JSON ─────────────────────────────────────────────────

def _find_result(report_dict: dict, keyword: str) -> dict:
    """
    Cherche le résultat de la première métrique dont le nom contient keyword.
    Compatible avec plusieurs versions d'Evidently (0.4.x, 0.5.x).
    """
    for m in report_dict.get("metrics", []):
        if keyword.lower() in str(m.get("metric", "")).lower():
            return m.get("result", {})
    return {}


def _nested(d: dict, *keys, default=None):
    """Accès imbriqué robuste : _nested(d, 'current', 'mae') sans KeyError."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _extract_quality(report_dict: dict) -> dict:
    """Extrait les métriques de qualité depuis le rapport Evidently."""
    out: dict = {}
    try:
        # DatasetMissingValuesMetric
        res  = _find_result(report_dict, "MissingValues")
        curr = res.get("current", res)
        if curr:
            out["missing_values_pct"] = round(
                float(curr.get("share_of_missing_values", 0)) * 100, 2
            )
            out["n_rows"]    = int(curr.get("number_of_rows", 0))
            out["n_missing"] = int(curr.get("number_of_missing_values", 0))

        # DatasetSummaryMetric — colonnes constantes, doublons
        res_sum  = _find_result(report_dict, "DatasetSummary")
        curr_sum = res_sum.get("current", res_sum)
        if curr_sum:
            out["n_duplicates"] = int(curr_sum.get("number_of_duplicated_rows", 0))
            out["n_constant"]   = int(curr_sum.get("number_of_constant_columns", 0))
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _extract_drift(report_dict: dict) -> dict:
    """Extrait les métriques de drift dataset + liste des features driftées."""
    out: dict = {}
    try:
        res  = _find_result(report_dict, "DatasetDrift")
        curr = res.get("current", res)

        # Drift global
        drift_flag = curr.get("dataset_drift", res.get("dataset_drift", None))
        if drift_flag is not None:
            out["dataset_drift"] = bool(drift_flag)

        share   = curr.get("share_of_drifted_columns",
                           res.get("share_of_drifted_columns", 0))
        n_drift = curr.get("number_of_drifted_columns",
                           res.get("number_of_drifted_columns", 0))
        n_cols  = curr.get("number_of_columns",
                           res.get("number_of_columns", 0))
        out["drift_share"] = round(float(share) * 100, 1)
        out["n_drifted"]   = int(n_drift)
        out["n_features"]  = int(n_cols)

        # Colonnes driftées individuellement
        drifted = []
        for m in report_dict.get("metrics", []):
            if "ColumnDrift" in str(m.get("metric", "")):
                r = m.get("result", {})
                if r.get("drift_detected", False):
                    drifted.append(r.get("column_name", "?"))
        if drifted:
            out["drifted_columns"] = drifted

    except Exception as exc:
        out["error"] = str(exc)
    return out


def _extract_performance(report_dict: dict) -> dict:
    """Extrait MAE, RMSE, MAPE courant vs référence + dégradation MAE %."""
    out: dict = {}
    try:
        res = _find_result(report_dict, "RegressionQuality")
        if res:
            for period in ("current", "reference"):
                p = res.get(period, {})
                out[f"mae_{period}"]  = round(float(p.get("mean_abs_error",      0)), 4)
                out[f"rmse_{period}"] = round(float(p.get("rmse",                0)), 4)
                out[f"mape_{period}"] = round(
                    float(p.get("mean_abs_perc_error", 0)) * 100, 2
                )

            # Dégradation MAE entre référence et courant (en %)
            mae_ref  = out.get("mae_reference", 0)
            mae_curr = out.get("mae_current",   0)
            if mae_ref > 0:
                out["mae_degradation_pct"] = round(
                    (mae_curr - mae_ref) / mae_ref * 100, 1
                )
    except Exception as exc:
        out["error"] = str(exc)
    return out
