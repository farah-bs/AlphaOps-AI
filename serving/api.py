import hashlib
import hmac
import os
import pickle
import threading
import warnings
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

try:
    RETRAIN_K = max(1, int(os.getenv("RETRAIN_K", "20")))
except ValueError:
    RETRAIN_K = 20

HMAC_SECRET = os.getenv("HMAC_SECRET", "alphaops-secret-change-me")
if HMAC_SECRET == "alphaops-secret-change-me":
    print("WARNING: HMAC_SECRET is set to the default value. Set a strong secret in production.")

_csv_lock = threading.Lock()

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
    # Use actual last known price from training history (not smoothed yhat)
    last_known_y   = float(model.history["y"].iloc[-1])

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

    with _csv_lock:
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


# ── Feedback email loop ────────────────────────────────────────────────────────

def _verify_token(ticker: str, pred_date: str, prediction: bool, token: str) -> bool:
    msg      = f"{ticker}:{pred_date}:{prediction}"
    expected = hmac.new(HMAC_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, token)


def _feedback_confirm_html(ticker: str, agreed: bool, n_prod: int, retrained: bool) -> str:
    action      = "validée ✅" if agreed else "contestée ❌"
    retrain_msg = "<p style='color:#f0b429'>🔄 Réentraînement déclenché automatiquement.</p>" if retrained else ""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AlphaOps — Merci</title>
</head>
<body style="margin:0;padding:0;background:#0d1117;font-family:Arial,sans-serif;color:#e6edf3;
             display:flex;align-items:center;justify-content:center;min-height:100vh">
  <div style="max-width:480px;text-align:center;padding:40px 24px">
    <div style="font-size:48px;margin-bottom:16px">📈</div>
    <h1 style="color:#00d4ff;margin-bottom:8px">AlphaOps AI</h1>
    <p style="font-size:18px;margin-bottom:4px">
      Analyse <strong style="color:#00d4ff">{ticker}</strong> — {action}
    </p>
    <p style="color:#8b949e;font-size:13px;margin-bottom:24px">
      Feedback enregistré ({n_prod} entrée(s) en production).
    </p>
    {retrain_msg}
    <p style="color:#484f58;font-size:12px;margin-top:32px">
      Merci pour votre retour. Il contribue à améliorer le modèle.
    </p>
  </div>
</body>
</html>"""


@app.get("/feedback/confirm", response_class=HTMLResponse)
def feedback_confirm(token: str, ticker: str, date: str, agreed: str, prediction: str):
    """
    Endpoint cliqué depuis le lien email.
    Vérifie le token HMAC, enregistre le feedback dans prod_data.csv,
    et déclenche un réentraînement si le seuil est atteint.
    """
    ticker = ticker.upper()

    if agreed.lower() not in ("true", "false"):
        return HTMLResponse("<h2>Paramètre invalide : agreed doit être true ou false.</h2>", status_code=400)
    if prediction.lower() not in ("true", "false"):
        return HTMLResponse("<h2>Paramètre invalide : prediction doit être true ou false.</h2>", status_code=400)

    agreed_bool     = agreed.lower() == "true"
    prediction_bool = prediction.lower() == "true"

    if not _verify_token(ticker, date, prediction_bool, token):
        return HTMLResponse("<h2>Lien invalide.</h2>", status_code=403)

    # target = prediction si accord, inverse sinon
    target = prediction_bool if agreed_bool else (not prediction_bool)

    row = {
        "symbol":     ticker,
        "date":       date,
        "target":     target,
        "prediction": prediction_bool,
        "agreed":     agreed_bool,
    }

    prod_path = DATA / "prod_data.csv"
    df_new    = pd.DataFrame([row])

    with _csv_lock:
        if prod_path.exists() and prod_path.stat().st_size > 0:
            df_new.to_csv(prod_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(prod_path, mode="w", header=True, index=False)
        n_prod = len(pd.read_csv(prod_path))

    retrained = False
    if n_prod % RETRAIN_K == 0:
        threading.Thread(target=_retrain_background, daemon=True).start()
        retrained = True

    return _feedback_confirm_html(ticker, agreed_bool, n_prod, retrained)
