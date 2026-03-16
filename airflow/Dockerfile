FROM apache/airflow:2.9.3

USER airflow
COPY requirements.txt /requirements.txt
# torch CPU séparé (index URL spécifique) avant le reste des requirements
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r /requirements.txt
