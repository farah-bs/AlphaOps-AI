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


def run_lstm_training():
    from ml.lstm.train_lstm import train_all_lstm
    from ml.training.config import LSTMConfig
    results = train_all_lstm(LSTMConfig())
    ok = [t for t, r in results.items() if "error" not in r]
    print(f"Modèles LSTM entraînés : {ok}")


with DAG(
    dag_id='lstm_entrainement',
    default_args=default_args,
    description='Entraîne les modèles LSTM de direction (J+1, J+7, J+30)',
    start_date=datetime(2026, 2, 1),
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'training', 'lstm'],
) as dag:

    train_task = PythonOperator(
        task_id='train_lstm_models',
        python_callable=run_lstm_training,
    )

    train_task
