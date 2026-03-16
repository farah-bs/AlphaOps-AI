"""
Inférence LSTM — AlphaOps AI
==============================
Charge un modèle LSTM entraîné et retourne les probabilités de hausse
à J+1, J+7 et J+30 pour un ticker donné.

Usage :
    from ml.lstm.predict_lstm import get_lstm_prediction
    result = get_lstm_prediction("AAPL")
    # {"prob_up_1d": 0.63, "prob_up_7d": 0.71, "prob_up_30d": 0.58, ...}
"""

import pickle
from pathlib import Path

ARTIFACTS = Path("/artifacts")          # chemin dans le container serving
_ROOT     = Path(__file__).resolve().parent.parent.parent
_ARTIFACTS_LOCAL = _ROOT / "artifacts"  # chemin local (dev / Airflow)


def _artifacts_dir() -> Path:
    """Retourne le bon chemin vers artifacts/ selon l'environnement."""
    return ARTIFACTS if ARTIFACTS.exists() else _ARTIFACTS_LOCAL


def get_lstm_prediction(ticker: str) -> dict:
    """
    Charge le modèle LSTM du ticker et calcule les probabilités de hausse.

    Args:
        ticker : symbole (ex: "AAPL")

    Returns:
        {
            "prob_up_1d":  float,   # P(prix J+1 > aujourd'hui)
            "prob_up_7d":  float,   # P(prix J+7 > aujourd'hui)
            "prob_up_30d": float,   # P(prix J+30 > aujourd'hui)
            "signal_1d":   str,     # "HAUSSE" / "BAISSE" / "NEUTRE"
            "signal_7d":   str,
            "signal_30d":  str,
            "horizons":    [1, 7, 30],
        }

    Raises:
        FileNotFoundError : modèle ou scaler absent (pas encore entraîné)
    """
    import torch
    from ml.features.feature_engineering import get_last_sequence
    from ml.lstm.model import build_model

    arts    = _artifacts_dir()
    pt_path = arts / f"lstm_{ticker}.pt"
    sc_path = arts / f"lstm_scaler_{ticker}.pickle"

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Modèle LSTM introuvable pour {ticker} ({pt_path}). "
            "Lancez d'abord le DAG lstm_training."
        )
    if not sc_path.exists():
        raise FileNotFoundError(f"Scaler LSTM introuvable : {sc_path}")

    # ── Chargement modèle ─────────────────────────────────────────────────────
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    model_cfg  = checkpoint["cfg"]
    model      = build_model(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        n_outputs=model_cfg["n_outputs"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    horizons = model_cfg.get("horizons", [1, 7, 30])

    # ── Chargement scaler ─────────────────────────────────────────────────────
    with open(sc_path, "rb") as f:
        scaler = pickle.load(f)

    # ── Dernière séquence ─────────────────────────────────────────────────────
    seq_len = model_cfg.get("seq_len", 60)
    # get_last_sequence retourne (1, seq_len, n_features) en float32
    x = get_last_sequence(ticker, seq_len=seq_len, scaler=scaler)

    # ── Inférence ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs  = torch.sigmoid(logits).squeeze(0).numpy()

    # probs shape : (n_outputs,) → [J+1, J+7, J+30]
    def _signal(p: float) -> str:
        if p >= 0.55:
            return "HAUSSE"
        if p <= 0.45:
            return "BAISSE"
        return "NEUTRE"

    keys   = ["prob_up_1d", "prob_up_7d", "prob_up_30d"]
    skeys  = ["signal_1d",  "signal_7d",  "signal_30d"]
    result = {"horizons": horizons}

    for i, (pk, sk) in enumerate(zip(keys, skeys)):
        p = float(probs[i]) if i < len(probs) else 0.5
        result[pk] = round(p, 4)
        result[sk] = _signal(p)

    return result
