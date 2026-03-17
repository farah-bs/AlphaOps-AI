from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


@dataclass
class TrainingConfig:
    # ── Data splits ───────────────────────────────────────────────────────────
    # None → dynamique : train_end = aujourd'hui - 7 jours
    #                    val_end   = aujourd'hui - 180 jours
    train_end: Optional[str] = None
    val_end:   Optional[str] = None

    # ── Daily model : fenêtre 60j → prédit J+1 ────────────────────────────────
    daily_window:  int = 60
    daily_horizon: int = 1

    # ── Monthly model : fenêtre 180j → prédit les 30 prochains jours ──────────
    monthly_window:  int = 180
    monthly_horizon: int = 30

    # ── Prophet hyperparamètres ───────────────────────────────────────────────
    n_changepoints:           int   = 15
    uncertainty_samples:      int   = 300
    interval_width:           float = 0.8
    changepoint_prior_scale:  float = 0.05
    seasonality_prior_scale:  float = 1.0
    seasonality_mode:         str   = "multiplicative"
    daily_seasonality:        bool  = False
    weekly_seasonality:       bool  = True
    yearly_seasonality:       bool  = True   # stocks have strong annual patterns

    series_gap_days:   int = 30
    retrain_threshold: int = 50

    # ── Tickers ───────────────────────────────────────────────────────────────
    tickers: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "SPY", "QQQ", "BTC-USD",
        ]
    )

    def __post_init__(self):
        today = datetime.now()
        if self.train_end is None:
            self.train_end = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        if self.val_end is None:
            self.val_end = (today - timedelta(days=180)).strftime("%Y-%m-%d")


@dataclass
class LSTMConfig:
    """Hyperparamètres pour le modèle LSTM de direction multi-horizon."""

    # ── Architecture ──────────────────────────────────────────────────────────
    input_size:  int   = 12    # = len(FEATURE_COLS) dans feature_engineering.py
    hidden_size: int   = 64
    num_layers:  int   = 2
    dropout:     float = 0.2

    # ── Séquences ─────────────────────────────────────────────────────────────
    seq_len:   int         = 60              # fenêtre de 60 jours en entrée
    horizons:  Tuple[int, ...] = (1, 7, 30) # horizons de prédiction (J+1, J+7, J+30)

    # ── Entraînement ──────────────────────────────────────────────────────────
    batch_size: int   = 64
    epochs:     int   = 100  # more headroom; early stopping will cut short if needed
    lr:         float = 1e-3
    patience:   int   = 15   # early stopping (val loss)

    # ── Data splits ───────────────────────────────────────────────────────────
    # Dynamiques : train | val | test découpés à partir d'aujourd'hui
    #   train : 2020-01-01 → aujourd'hui - 365j
    #   val   : aujourd'hui - 365j → aujourd'hui - 90j
    #   test  : aujourd'hui - 90j  → aujourd'hui - 7j (non utilisé en prod)
    train_end: Optional[str] = None   # None → dynamique
    val_end:   Optional[str] = None   # None → dynamique

    # ── Tickers ───────────────────────────────────────────────────────────────
    tickers: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "SPY", "QQQ", "BTC-USD",
        ]
    )

    def __post_init__(self):
        today = datetime.now()
        if self.train_end is None:
            self.train_end = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        if self.val_end is None:
            self.val_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
