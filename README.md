# AlphaOps AI

AlphaOps AI is an end-to-end production-oriented AI system combining 
Data Engineering, MLOps, LLM-powered SQL Agents, and autonomous workflows.

The project demonstrates how to design, deploy, monitor, and continuously 
improve a machine learning system in a real-world financial context.

## Architecture Overview

### 1️⃣ Data Engineering Layer (✔️ Done)
- Daily stock ingestion via yfinance
- PostgreSQL Star Schema (DimTickers, DimTime, FactOHLCV)
- Batch orchestration with Apache Airflow

### 2️⃣ LLM SQL Agent (✔️ Done)
- LangChain + Mistral Codestral
- Natural language → validated SQL (SELECT-only)
- Schema-grounded RAG
- Secure SQL validation layer

### 3️⃣ ML Forecasting Layer (✔️ Done)
- LSTM time-series prediction model
- Feature engineering (returns, volatility, rolling metrics)
- Model versioning
- Performance tracking

### 4️⃣ MLOps & Monitoring (✔️ Done)
- Model serving with FastAPI
- User dashboard via Streamlit
- Automated monitoring (performance, drift, quality) using Evidently
- Metrics storage in PostgreSQL

### 5️⃣ Autonomous AI Agent (✔️ Done)
Orchestrated via n8n + LLM:
- Sends automated email notifications
- Explains model predictions
- Collects user feedback
- Analyzes feedback using an LLM
- Triggers retraining when thresholds are reached

### 6️⃣ Infrastructure (✔️ Done)
- Fully containerized with Docker Compose
- Modular microservices architecture
- CI/CD ready structure

---

## Key Objective

Build a production-grade AI system that bridges:

Data Engineering × LLM Agents × MLOps × Autonomous Decision Systems