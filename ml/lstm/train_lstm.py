"""
Pipeline d'entraînement LSTM — AlphaOps AI
===========================================
Entraîne un LSTMDirectionModel par ticker pour prédire la direction
du prix à J+1, J+7 et J+30.

Fonctionnalités :
    - Early stopping sur la val loss (patience configurable)
    - pos_weight calculé sur le train pour corriger le class imbalance
    - Sauvegarde du meilleur modèle (val loss minimale) + scaler sklearn
    - Logging MLflow : params, métriques epoch par epoch, artifacts

Usage :
    from ml.lstm.train_lstm import train_all_lstm
    from ml.training.config import LSTMConfig
    train_all_lstm(LSTMConfig())
"""

import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


# ── Entraînement d'un ticker ───────────────────────────────────────────────────

def train_lstm_ticker(ticker: str, cfg) -> dict:
    """
    Entraîne le LSTM pour un ticker et sauvegarde les artifacts.

    Returns:
        dict avec val_loss, val_acc_1d, val_acc_7d, val_acc_30d
    """
    import torch
    import torch.nn as nn

    from ml.features.feature_engineering import prepare_data_lstm
    from ml.lstm.dataset import build_loaders
    from ml.lstm.model import build_model

    print(f"\n  [{ticker}] Préparation des données...")
    (X_train, y_train), (X_val, y_val), _, scaler = prepare_data_lstm(
        ticker=ticker,
        seq_len=cfg.seq_len,
        train_end=cfg.train_end,
        val_end=cfg.val_end,
        horizons=cfg.horizons,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(f"[{ticker}] Séquences insuffisantes après split.")

    print(f"  [{ticker}] Train: {len(X_train)} séq | Val: {len(X_val)} séq")

    # ── pos_weight : corrige le class imbalance par horizon ───────────────────
    # pos_weight[h] = nb_négatifs / nb_positifs sur le train
    pos_weight = torch.tensor(
        [
            (y_train[:, h] == 0).sum() / max((y_train[:, h] == 1).sum(), 1)
            for h in range(len(cfg.horizons))
        ],
        dtype=torch.float32,
    )

    train_loader, val_loader = build_loaders(
        X_train, y_train, X_val, y_val, batch_size=cfg.batch_size
    )

    # ── Modèle, loss, optimiseur ──────────────────────────────────────────────
    model     = build_model(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        n_outputs=len(cfg.horizons),
    )

    def focal_loss(logits, targets, gamma=2.0):
        import torch.nn.functional as F
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** gamma * bce).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Boucle d'entraînement avec early stopping ─────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history        = []

    print(f"  [{ticker}] Entraînement ({cfg.epochs} epochs max, patience={cfg.patience})...")

    for epoch in range(1, cfg.epochs + 1):
        # — Train —
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = focal_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # — Val —
        model.eval()
        val_loss = 0.0
        all_probs, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += focal_loss(logits, yb).item() * len(xb)
                all_probs.append(torch.sigmoid(logits).numpy())
                all_y.append(yb.numpy())
        val_loss /= len(X_val)

        all_probs = np.vstack(all_probs)
        all_y     = np.vstack(all_y)
        val_accs  = ((all_probs > 0.5).astype(int) == all_y.astype(int)).mean(axis=0)

        scheduler.step(val_loss)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc_1d": val_accs[0], "val_acc_7d": val_accs[1], "val_acc_30d": val_accs[2],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} "
                f"| acc J+1={val_accs[0]:.3f} J+7={val_accs[1]:.3f} J+30={val_accs[2]:.3f}"
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"    Early stopping à l'epoch {epoch}")
                break

    # ── Sauvegarde du meilleur modèle ─────────────────────────────────────────
    model.load_state_dict(best_state)

    pt_path     = ARTIFACTS / f"lstm_{ticker}.pt"
    scaler_path = ARTIFACTS / f"lstm_scaler_{ticker}.pickle"

    torch.save({
        "state_dict":  model.state_dict(),
        "cfg": {
            "input_size":  cfg.input_size,
            "hidden_size": cfg.hidden_size,
            "num_layers":  cfg.num_layers,
            "dropout":     cfg.dropout,
            "n_outputs":   len(cfg.horizons),
            "horizons":    list(cfg.horizons),
            "seq_len":     cfg.seq_len,
        },
    }, pt_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    best_hist = history[-patience_count - 1] if patience_count else history[-1]
    print(
        f"  [{ticker}] ✓ best val_loss={best_hist['val_loss']:.4f} "
        f"| J+1={best_hist['val_acc_1d']:.3f} "
        f"| J+7={best_hist['val_acc_7d']:.3f} "
        f"| J+30={best_hist['val_acc_30d']:.3f}"
    )

    return {
        "ticker":      ticker,
        "val_loss":    best_hist["val_loss"],
        "val_acc_1d":  best_hist["val_acc_1d"],
        "val_acc_7d":  best_hist["val_acc_7d"],
        "val_acc_30d": best_hist["val_acc_30d"],
        "n_epochs":    best_hist["epoch"],
        "history":     history,
    }


# ── Entraînement multi-tickers ─────────────────────────────────────────────────

def train_all_lstm(cfg=None) -> dict:
    """
    Entraîne le LSTM sur tous les tickers définis dans LSTMConfig.
    Séquentiel (PyTorch CPU — ProcessPool peu utile, Stan non impliqué).
    Log les résultats dans MLflow.

    Returns:
        dict ticker → métriques
    """
    from ml.training.config import LSTMConfig
    if cfg is None:
        cfg = LSTMConfig()

    print(f"\n  [LSTM Config] train_end={cfg.train_end} | "
          f"hidden={cfg.hidden_size} | layers={cfg.num_layers} | "
          f"epochs={cfg.epochs} | horizons={cfg.horizons}\n")

    results = {}
    for ticker in cfg.tickers:
        try:
            res = train_lstm_ticker(ticker, cfg)
            results[ticker] = res
        except Exception as e:
            print(f"  [{ticker}] ERREUR — {e}")
            results[ticker] = {"error": str(e)}

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Résumé LSTM")
    print("=" * 55)
    for ticker, res in results.items():
        if "error" in res:
            print(f"  {ticker:<10} ERREUR : {res['error']}")
        else:
            print(
                f"  {ticker:<10} val_loss={res['val_loss']:.4f} | "
                f"J+1={res['val_acc_1d']:.3f} | "
                f"J+7={res['val_acc_7d']:.3f} | "
                f"J+30={res['val_acc_30d']:.3f} | "
                f"epochs={res['n_epochs']}"
            )

    # ── Logging MLflow ────────────────────────────────────────────────────────
    try:
        _log_lstm_to_mlflow(cfg, results)
    except Exception as e:
        print(f"  [MLflow] Avertissement : {e}")

    return results


# ── Logging MLflow ─────────────────────────────────────────────────────────────

def _log_lstm_to_mlflow(cfg, results: dict) -> None:
    """Log un run MLflow pour l'entraînement LSTM (params + métriques agrégées)."""
    import mlflow
    from ml.registry.mlflow_utils import (
        get_or_create_experiment, log_params_safe,
        log_metrics_safe, log_artifact_path,
        make_run_name, setup_mlflow,
    )

    EXPERIMENT_LSTM = "AlphaOps-LSTM-Training"

    setup_mlflow()
    exp_id   = get_or_create_experiment(EXPERIMENT_LSTM)
    run_name = make_run_name("train_lstm")

    ok = {t: r for t, r in results.items() if "error" not in r}

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags({
            "project":      "AlphaOps AI",
            "model_family": "lstm",
            "stage":        "training",
        })

        log_params_safe({
            "tickers":      cfg.tickers,
            "train_end":    cfg.train_end,
            "val_end":      cfg.val_end,
            "horizons":     list(cfg.horizons),
            "seq_len":      cfg.seq_len,
            "hidden_size":  cfg.hidden_size,
            "num_layers":   cfg.num_layers,
            "dropout":      cfg.dropout,
            "batch_size":   cfg.batch_size,
            "epochs":       cfg.epochs,
            "lr":           cfg.lr,
            "patience":     cfg.patience,
        })

        if ok:
            log_metrics_safe({
                "n_trained":       len(ok),
                "mean_val_loss":   np.mean([r["val_loss"]  for r in ok.values()]),
                "mean_acc_1d":     np.mean([r["val_acc_1d"] for r in ok.values()]),
                "mean_acc_7d":     np.mean([r["val_acc_7d"] for r in ok.values()]),
                "mean_acc_30d":    np.mean([r["val_acc_30d"] for r in ok.values()]),
            })
            for ticker, r in ok.items():
                log_metrics_safe({
                    f"{ticker}_val_loss":  r["val_loss"],
                    f"{ticker}_acc_1d":    r["val_acc_1d"],
                    f"{ticker}_acc_7d":    r["val_acc_7d"],
                    f"{ticker}_acc_30d":   r["val_acc_30d"],
                })

        # Log des fichiers .pt comme artifacts
        for ticker in ok:
            log_artifact_path(ARTIFACTS / f"lstm_{ticker}.pt")
            log_artifact_path(ARTIFACTS / f"lstm_scaler_{ticker}.pickle")

    print("  [MLflow] Run LSTM loggué.")
