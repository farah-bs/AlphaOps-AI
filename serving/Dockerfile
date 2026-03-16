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
# torch CPU séparé (index URL spécifique) avant le reste des requirements
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip3 install --no-cache-dir -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port 8080
