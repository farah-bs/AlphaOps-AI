from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')
from data.fetch_live_stocks import run_daily_batch

default_args = {
    'owner': 'adam',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_stock_ingestion',
    default_args=default_args,
    description='Ingest daily stock data into the database',
    start_date=datetime(2026, 2, 1),
    schedule_interval='0 18 * * 1-5',
    catchup=False,
    max_active_runs=1,
    tags=['stocks', 'finance', 'postgres']
) as dag:
    
    ingest_task = PythonOperator(
        task_id='ingest_incremental',
        python_callable=run_daily_batch
    )

    ingest_task