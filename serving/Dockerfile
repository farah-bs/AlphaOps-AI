FROM python:3.11-slim

WORKDIR /app

# Contexte de build = racine du projet (défini dans docker-compose.yml)
COPY serving/api.py .
COPY serving/requirements.txt .

# Code ML nécessaire pour le réentraînement en arrière-plan
COPY ml ./ml
COPY scripts ./scripts
COPY src ./src

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port 8080
