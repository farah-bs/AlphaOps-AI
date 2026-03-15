FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "file:///mlruns", \
     "--artifacts-destination", "file:///mlruns/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "*"]
