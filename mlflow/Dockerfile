FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow
EXPOSE 5000
# umask 000 : tous les dossiers/fichiers créés par MLflow server seront en 777/666
# → le user airflow (uid 50000) peut écrire dans /mlruns sans être root
CMD ["sh", "-c", \
     "umask 000 && \
      mkdir -p /mlruns/artifacts && \
      chmod -R 777 /mlruns && \
      exec mlflow server \
        --backend-store-uri file:///mlruns \
        --default-artifact-root mlflow-artifacts:/ \
        --artifacts-destination file:///mlruns/artifacts \
        --serve-artifacts \
        --host 0.0.0.0 \
        --port 5000 \
        --allowed-hosts '*'"]
