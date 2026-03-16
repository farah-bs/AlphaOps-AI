import pickle
import warnings
import threading
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import norm
from typing import Literal

warnings.filterwarnings("ignore")

ARTIFACTS = Path("/artifacts")
DATA      = Path("/data")

# ── Variables globales — chargées au démarrage ────────────────────────────────
models_daily:   dict = {}
models_monthly: dict = {}
price_stats:    dict = {}

TRAIN_END = pd.Timestamp("2023-12-31")
RETRAIN_K = 50

# Tickers qui tradent 7j/7 → fréquence calendaire au lieu des jours ouvrés
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}

app = FastAPI(title="AlphaOps Serving API")


@app.on_event("startup")
def load_artifacts():
    global models_daily, models_monthly, price_stats
    models_daily   = pickle.loads((ARTIFACTS / "models_daily.pickle").read_bytes())
    models_monthly = pickle.loads((ARTIFACTS / "models_monthly.pickle").read_bytes())
    price_stats    = pickle.loads((ARTIFACTS / "price_stats.pickle").read_bytes())
    print(f"Modèles chargés : {list(models_daily.keys())}")


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker: str
    mode:   Literal["daily", "monthly"] = "daily"

class LSTMPredictRequest(BaseModel):
    ticker: str

class FeedbackRequest(BaseModel):
    ticker:     str
    date:       str
    features:   dict
    prediction: bool
    target:     bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(models_daily.keys()),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    ticker = req.ticker.upper()

    if req.mode == "daily":
        model   = models_daily.get(ticker)
        horizon = 1
    else:
        model   = models_monthly.get(ticker)
        horizon = 30

    if model is None:
        raise HTTPException(status_code=404, detail=f"Aucun modèle pour {ticker}")

    # Vraie dernière date du modèle → on étend jusqu'à aujourd'hui + horizon
    today         = pd.Timestamp.today().normalize()
    last_train_dt = pd.Timestamp(model.history_dates.max()).normalize()
    since         = last_train_dt + pd.Timedelta(days=1)

    if ticker in CRYPTO_TICKERS:
        days_since     = len(pd.date_range(since, today))
        days_to_extend = max(days_since + horizon, horizon)
        future         = model.make_future_dataframe(periods=days_to_extend, freq="D")
    else:
        days_since     = len(pd.bdate_range(since, today))
        days_to_extend = max(days_since + horizon, horizon)
        future         = model.make_future_dataframe(periods=days_to_extend, freq="B")

    forecast = model.predict(future)

    # Les `horizon` dernières lignes = prédictions futures
    preds          = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    last_known_y   = float(forecast.iloc[-(horizon + 1)]["yhat"])

    # ── Calcul prob_up ────────────────────────────────────────────────────────
    # Prophet génère des CI symétriques autour de yhat.
    # On approxime la distribution comme normale :
    #   σ = (yhat_upper - yhat_lower) / (2 × z_α/2)
    # où z_α/2 correspond à l'interval_width du modèle (ex: 0.8 → z=1.28).
    # prob_up = P(prix prédit > dernier prix connu)
    iw     = getattr(model, "interval_width", 0.8)
    z_half = norm.ppf((1.0 + iw) / 2.0)  # ex: iw=0.8 → z=1.2816

    predictions = []
    for _, row in preds.iterrows():
        width = float(row["yhat_upper"]) - float(row["yhat_lower"])
        sigma = width / (2.0 * z_half) if z_half > 0 else 0.0

        if sigma > 0:
            # P(X > last_known_y) où X ~ N(yhat, sigma)
            prob_up = float(norm.sf(last_known_y, loc=float(row["yhat"]), scale=sigma))
        else:
            prob_up = 1.0 if float(row["yhat"]) > last_known_y else 0.0

        predictions.append({
            "date":       row["ds"].strftime("%Y-%m-%d"),
            "yhat":       round(float(row["yhat"]),       2),
            "yhat_lower": round(float(row["yhat_lower"]), 2),
            "yhat_upper": round(float(row["yhat_upper"]), 2),
            "prob_up":    round(prob_up, 4),
        })

    direction = predictions[-1]["prob_up"] > 0.5

    return {
        "ticker":      ticker,
        "mode":        req.mode,
        "direction":   direction,
        "predictions": predictions,
    }


@app.post("/predict/lstm")
def predict_lstm(req: LSTMPredictRequest):
    """
    Probabilités de hausse LSTM à J+1, J+7 et J+30.

    Retourne :
        prob_up_1d  : P(prix dans 1 jour > prix aujourd'hui)
        prob_up_7d  : P(prix dans 7 jours > prix aujourd'hui)
        prob_up_30d : P(prix dans 30 jours > prix aujourd'hui)
        signal_*    : "HAUSSE" (>55%) / "BAISSE" (<45%) / "NEUTRE"
    """
    ticker = req.ticker.upper()
    try:
        from ml.lstm.predict_lstm import get_lstm_prediction
        result = get_lstm_prediction(ticker)
        return {"ticker": ticker, **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LSTM : {e}")


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    row = {
        "symbol":     req.ticker.upper(),
        "date":       req.date,
        **req.features,
        "target":     req.target,
        "prediction": req.prediction,
    }

    prod_path = DATA / "prod_data.csv"
    df_new    = pd.DataFrame([row])

    if prod_path.exists() and prod_path.stat().st_size > 0:
        df_new.to_csv(prod_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(prod_path, mode="w", header=True, index=False)

    n_prod = len(pd.read_csv(prod_path))

    retrained = False
    if n_prod % RETRAIN_K == 0:
        threading.Thread(target=_retrain_background, daemon=True).start()
        retrained = True

    return {"status": "ok", "n_prod": n_prod, "retrained": retrained}


def _retrain_background():
    """Lance le réentraînement dans un thread séparé pour ne pas bloquer l'API."""
    global models_daily, models_monthly
    import sys
    sys.path.insert(0, "/app")
    try:
        from scripts.retrain import retrain
        result = retrain()
        models_daily.update(result["models_daily"])
        models_monthly.update(result["models_monthly"])
        print("Modèles mis à jour après réentraînement.")
    except Exception as e:
        print(f"Erreur lors du réentraînement : {e}")
