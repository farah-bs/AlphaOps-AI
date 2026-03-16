from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class TrainingConfig:
    # ── Data splits ───────────────────────────────────────────────────────────
    # train_end = None → dynamique : aujourd'hui - 7 jours
    # Permet d'éviter une extrapolation de centaines de jours lors du serving
    train_end: Optional[str] = None
    val_end:   str = "2024-06-30"

    # ── Daily model : fenêtre 60j → prédit J+1 ────────────────────────────────
    daily_window:  int = 60
    daily_horizon: int = 1

    # ── Monthly model : fenêtre 180j → prédit les 30 prochains jours ──────────
    monthly_window:  int = 180
    monthly_horizon: int = 30

    # ── Prophet hyperparamètres ───────────────────────────────────────────────
    # n_changepoints réduit (25 → 15) : fit 30% plus rapide
    n_changepoints:           int   = 15
    # uncertainty_samples réduit (1000 → 300) : predict 3× plus rapide
    uncertainty_samples:      int   = 300
    # interval_width : largeur des intervalles de confiance (80%)
    interval_width:           float = 0.8
    # changepoint_prior_scale : réactivité aux ruptures de tendance
    changepoint_prior_scale:  float = 0.05
    # seasonality_prior_scale réduit (10 → 1) : évite le sur-ajustement saisonnier
    seasonality_prior_scale:  float = 1.0
    # multiplicative : variation en % stable → mieux pour les actions/crypto
    seasonality_mode:         str   = "multiplicative"
    daily_seasonality:        bool  = False
    weekly_seasonality:       bool  = True
    yearly_seasonality:       bool  = False

    # Jours de gap entre deux séries dans le df d'entraînement
    series_gap_days: int = 30

    # ── Trigger réentraînement ────────────────────────────────────────────────
    retrain_threshold: int = 50

    # ── Tickers ───────────────────────────────────────────────────────────────
    tickers: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "SPY", "QQQ", "BTC-USD",
        ]
    )

    def __post_init__(self):
        # train_end dynamique : 7 jours de marge pour garantir que la DB est à jour
        if self.train_end is None:
            self.train_end = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
