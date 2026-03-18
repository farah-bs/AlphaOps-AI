from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')

default_args = {
    'owner': 'adam',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def run_training():
    from ml.training.train import train
    result = train()
    print(f"Modèles entraînés : {list(result['models_daily'].keys())}")


with DAG(
    dag_id='prophet_entrainement',
    default_args=default_args,
    description='Entraîne les modèles FB Prophet (daily J+1 / monthly J+30) et sauvegarde les artifacts',
    start_date=datetime(2026, 2, 1),
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'training', 'prophet'],
) as dag:

    train_task = PythonOperator(
        task_id='train_models',
        python_callable=run_training,
    )

    train_task
