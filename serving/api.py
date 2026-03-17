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
from typing import Literal

warnings.filterwarnings("ignore")

ARTIFACTS = Path("/artifacts")
DATA      = Path("/data")

# ── Variables globales — chargées au démarrage ────────────────────────────────
models_daily:   dict = {}
models_monthly: dict = {}
price_stats:    dict = {}

TRAIN_END   = pd.Timestamp("2023-12-31")
RETRAIN_K   = int(os.getenv("RETRAIN_K", "20"))
HMAC_SECRET = os.getenv("HMAC_SECRET", "alphaops-secret-change-me")

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

class FeedbackRequest(BaseModel):
    ticker:     str
    date:       str   
    features:   dict  
    prediction: bool  # 
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

    # Vraie dernière date du modèle (fenêtres non-overlapping peuvent s'arrêter
    # avant TRAIN_END si le reste < window est ignoré)
    today          = pd.Timestamp.today().normalize()
    last_train_dt  = pd.Timestamp(model.history_dates.max()).normalize()
    business_days  = len(pd.bdate_range(last_train_dt + pd.Timedelta(days=1), today))
    days_to_extend = max(business_days + horizon, horizon)

    future   = model.make_future_dataframe(periods=days_to_extend, freq="B")
    forecast = model.predict(future)

    # Les `horizon` dernières lignes = prédictions futures
    preds         = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    last_known_y  = forecast.iloc[-(horizon + 1)]["yhat"]
    direction     = bool(preds.iloc[-1]["yhat"] > last_known_y)

    return {
        "ticker":    ticker,
        "mode":      req.mode,
        "direction": direction,
        "predictions": [
            {
                "date":       row["ds"].strftime("%Y-%m-%d"),
                "yhat":       round(float(row["yhat"]),       4),
                "yhat_lower": round(float(row["yhat_lower"]), 4),
                "yhat_upper": round(float(row["yhat_upper"]), 4),
            }
            for _, row in preds.iterrows()
        ],
    }


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
    ticker     = ticker.upper()
    agreed_bool     = agreed.lower() == "true"
    prediction_bool = prediction.lower() == "true"

    if not _verify_token(ticker, date, prediction_bool, token):
        return HTMLResponse("<h2>Lien invalide ou expiré.</h2>", status_code=403)

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

    if prod_path.exists() and prod_path.stat().st_size > 0:
        df_new.to_csv(prod_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(prod_path, mode="w", header=True, index=False)

    n_prod    = len(pd.read_csv(prod_path))
    retrained = False
    if n_prod % RETRAIN_K == 0:
        threading.Thread(target=_retrain_background, daemon=True).start()
        retrained = True

    return _feedback_confirm_html(ticker, agreed_bool, n_prod, retrained)
