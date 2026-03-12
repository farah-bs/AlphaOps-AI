from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # Data splits
    train_end: str = "2023-12-31"
    val_end:   str = "2024-06-30"

    # ── Daily model : fenêtre 60j → prédit J+1 ────────────────────────────
    daily_window:  int = 60
    daily_horizon: int = 1

    # ── Monthly model : fenêtre 180j → prédit les 30 prochains jours ──────
    monthly_window:  int = 180
    monthly_horizon: int = 30  

    # ── FBProphet hyperparamètres ─────────────────
    changepoint_prior_scale:  float = 0.05
    seasonality_prior_scale:  float = 10.0
    seasonality_mode:         str   = "additive"
    daily_seasonality:        bool  = False
    weekly_seasonality:       bool  = True
    yearly_seasonality:       bool  = False 

    # Jours de gap entre deux séries de tickers dans le df d'entraînement
    series_gap_days: int = 30

    # ── Trigger réentraînement ─────────────────────────────────────────────
    retrain_threshold: int = 50   # toutes les 50 nouvelles données prod

    # ── Tickers ────────────────────────────────────────────────────────────
    tickers: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "SPY", "QQQ", "BTC-USD",
        ]
    )
