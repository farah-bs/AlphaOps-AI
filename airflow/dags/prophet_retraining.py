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


def run_prophet_retraining():
    """
    Réentraîne les modèles FB Prophet sur les données fraîches de la DB.
    Met à jour les pickles models_daily.pickle et models_monthly.pickle.
    """
    from scripts.retrain import retrain
    result = retrain()
    n_daily   = len(result['models_daily'])
    n_monthly = len(result['models_monthly'])
    print(f"Réentraînement terminé — {n_daily} modèles daily, {n_monthly} modèles monthly")


with DAG(
    dag_id='prophet_reentrainement',
    default_args=default_args,
    description='Réentraîne les modèles FB Prophet sur données fraîches — chaque dimanche 20h',
    start_date=datetime(2026, 2, 1),
    schedule_interval='0 20 * * 0',  # chaque dimanche à 20h
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'prophet', 'retrain'],
) as dag:

    retrain_task = PythonOperator(
        task_id='retrain_prophet_models',
        python_callable=run_prophet_retraining,
        execution_timeout=timedelta(hours=2),
    )

    retrain_task
