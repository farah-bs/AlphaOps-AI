from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')

default_args = {
    'owner': 'adam',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD"]


def run_monitoring_all():
    """Lance le monitoring Evidently pour tous les tickers."""
    from ml.monitoring.monitor import run_monitoring

    for ticker in TICKERS:
        print(f"\n{'─' * 50}")
        print(f"  Monitoring : {ticker}")
        print(f"{'─' * 50}")
        try:
            summary = run_monitoring(ticker)
            drift   = summary["data_drift"]
            perf    = summary["model_performance"]
            quality = summary["data_quality"]

            # Log résumé dans les logs Airflow
            if "error" not in drift:
                print(
                    f"  Drift : {'🔴 DÉTECTÉ' if drift.get('dataset_drift') else '🟢 OK'}"
                    f" — {drift.get('n_drifted', '?')}/{drift.get('n_features', '?')} features"
                )
            if "error" not in quality:
                print(f"  Qualité : {quality.get('missing_values_pct', '?')}% manquants")
            if "error" not in perf:
                deg = perf.get("mae_degradation_pct")
                if deg is not None:
                    sign = "+" if deg > 0 else ""
                    print(f"  Performance : MAE {sign}{deg:.1f}% vs référence")
        except Exception as exc:
            print(f"  ERREUR pour {ticker} : {exc}")


with DAG(
    dag_id='evidently_monitoring',
    default_args=default_args,
    description='Monitoring automatisé — Evidently (qualité, drift, performance) — lun–ven 18h',
    start_date=datetime(2026, 1, 1),
    schedule_interval='0 18 * * 1-5',   # lun–ven à 18h (après clôture marché)
    catchup=False,
    tags=['ml', 'monitoring', 'evidently'],
) as dag:

    monitoring_task = PythonOperator(
        task_id='run_monitoring_all_tickers',
        python_callable=run_monitoring_all,
    )

    monitoring_task
