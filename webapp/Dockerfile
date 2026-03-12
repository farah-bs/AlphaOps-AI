FROM python:3.10-slim

WORKDIR /app

# Contexte de build = racine du projet
COPY webapp/requirements.txt .
COPY webapp/app.py .
COPY src ./src

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD streamlit run app.py --server.port 8501 --server.address 0.0.0.0
