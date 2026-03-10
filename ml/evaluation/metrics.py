"""Métriques d'évaluation pour les modèles Prophet stock"""
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% de mouvements haut/bas correctement prédits."""
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    return float(np.mean(true_dir == pred_dir))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Métriques de classification sur la direction J+1.
    y_true / y_pred : tableaux de prix → la direction est calculée internement.
    """
    true_dir = (np.diff(y_true) > 0).astype(int)
    pred_dir = (np.diff(y_pred) > 0).astype(int)
    return {
        "f1":                   f1_score(true_dir, pred_dir, zero_division=0),
        "precision":            precision_score(true_dir, pred_dir, zero_division=0),
        "recall":               recall_score(true_dir, pred_dir, zero_division=0),
        "balanced_accuracy":    balanced_accuracy_score(true_dir, pred_dir),
        "directional_accuracy": float(np.mean(true_dir == pred_dir)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, MAPE sur les prix."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate_prophet(model, df_val, horizon: int) -> dict:
    """
    Évalue un modèle Prophet sur df_val (format Prophet : ds, y).
    Retourne les métriques de régression + classification de direction.

    Args:
        model   : modèle Prophet déjà entraîné
        df_val  : DataFrame avec colonnes 'ds' et 'y' (données de validation)
        horizon : nombre de jours à prédire au-delà des données connues
    """
    future      = model.make_future_dataframe(periods=horizon, freq="B")
    predictions = model.predict(future)

    pred_aligned = (
        predictions[predictions["ds"].isin(df_val["ds"])]["yhat"].values
    )
    true_aligned = df_val["y"].values[: len(pred_aligned)]

    reg = regression_metrics(true_aligned, pred_aligned)
    clf = classification_metrics(true_aligned, pred_aligned) if len(true_aligned) > 1 else {}
    return {**reg, **clf}


def print_metrics(metrics: dict, label: str = "") -> None:
    if label:
        print(f"\n{'─' * 40}\n  {label}\n{'─' * 40}")
    for k, v in metrics.items():
        print(f"  {k:<24}: {v:.4f}")
