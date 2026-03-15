"""
MLflow Utilities — AlphaOps AI
================================
Module central de configuration et d'helpers MLflow.

Toute la config MLflow passe par ici. Les autres modules importent
depuis ce fichier, jamais depuis mlflow directement (sauf start_run).

Variables d'environnement supportées :
    MLFLOW_TRACKING_URI  URI du tracking server MLflow.
                         Défaut : file://<racine_projet>/mlruns (local)

Expériences créées :
    AlphaOps-Prophet-Training   runs d'entraînement Prophet
    AlphaOps-Prophet-Backtest   runs de rolling backtest walk-forward

Usage :
    from ml.registry.mlflow_utils import (
        EXPERIMENT_TRAINING, EXPERIMENT_BACKTEST, BASE_TAGS,
        setup_mlflow, get_or_create_experiment, make_run_name,
        log_params_safe, log_metrics_safe, log_artifact_path,
        log_dict_as_artifact,
    )
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow

log = logging.getLogger(__name__)

# ── Noms des expériences MLflow ────────────────────────────────────────────────
EXPERIMENT_TRAINING = "AlphaOps-Prophet-Training"
EXPERIMENT_BACKTEST = "AlphaOps-Prophet-Backtest"

# Tags communs à tous les runs du projet
BASE_TAGS: Dict[str, str] = {
    "project":      "AlphaOps AI",
    "model_family": "prophet",
}

# Chemin fallback local (racine projet) si MLFLOW_TRACKING_URI est absent
_ROOT          = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MLRUNS = _ROOT / "mlruns"


# ── Configuration ──────────────────────────────────────────────────────────────

def get_tracking_uri() -> str:
    """
    Retourne la tracking URI MLflow à utiliser.

    Priorité :
        1. Variable d'environnement MLFLOW_TRACKING_URI
        2. Fallback : dossier ./mlruns à la racine du projet (usage local)
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    # Path.as_uri() gère les espaces (%20) et les backslashes Windows proprement
    return _DEFAULT_MLRUNS.as_uri()


def setup_mlflow() -> None:
    """
    Configure la tracking URI MLflow.
    À appeler une fois avant tout start_run ou get_experiment.
    Idempotent : peut être appelé plusieurs fois sans effet de bord.
    """
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    log.debug(f"[MLflow] Tracking URI : {uri}")


def get_or_create_experiment(name: str) -> str:
    """
    Retourne l'ID de l'expérience MLflow, la crée si elle n'existe pas encore.

    Args:
        name : nom de l'expérience (utiliser EXPERIMENT_TRAINING ou EXPERIMENT_BACKTEST)

    Returns:
        experiment_id (str)
    """
    setup_mlflow()
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        log.info(f"[MLflow] Expérience créée : '{name}' (id={exp_id})")
    else:
        exp_id = exp.experiment_id
    return exp_id


# ── Nommage des runs ───────────────────────────────────────────────────────────

def make_run_name(prefix: str, ticker: Optional[str] = None) -> str:
    """
    Génère un nom de run MLflow lisible et horodaté.

    Exemples :
        make_run_name("train_prophet")          → "train_prophet_2026-03-12"
        make_run_name("backtest", "AAPL")       → "backtest_AAPL_2026-03-12"
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if ticker:
        return f"{prefix}_{ticker}_{today}"
    return f"{prefix}_{today}"


# ── Helpers de logging ─────────────────────────────────────────────────────────

def log_params_safe(params: Dict[str, Any]) -> None:
    """
    Log des paramètres MLflow de façon robuste.

    Convertit listes et types non standard en strings.
    Ignore silencieusement les valeurs non sérialisables.
    """
    safe: Dict[str, Any] = {}
    for k, v in params.items():
        try:
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, (list, tuple)):
                # Liste → "AAPL,MSFT,GOOGL"
                safe[k] = ",".join(str(x) for x in v)
            else:
                safe[k] = str(v)
        except Exception:
            safe[k] = "<non sérialisable>"
    mlflow.log_params(safe)


def log_metrics_safe(metrics: Dict[str, Any]) -> None:
    """
    Log des métriques MLflow (doit être numérique).

    Ignore silencieusement les valeurs non numériques avec un warning.
    """
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except (TypeError, ValueError):
            log.warning(f"[MLflow] Métrique '{k}' ignorée (non numérique) : {v!r}")


def log_artifact_path(path: Path) -> None:
    """
    Log un fichier comme artifact MLflow.
    Affiche un message si le fichier est introuvable ou si le logging échoue.
    """
    if not path.exists():
        print(f"  [MLflow] Artifact introuvable, ignoré : {path}")
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception as e:
        print(f"  [MLflow] Échec artifact '{path.name}' : {e}")


def log_dict_as_artifact(data: Dict, filename: str, artifact_path: str = "summaries") -> None:
    """
    Sérialise un dict en JSON et le logue comme artifact MLflow.

    Utilise un fichier temporaire pour éviter de polluer le workspace.

    Args:
        data          : dict à sérialiser
        filename      : nom de base du fichier (sans extension)
        artifact_path : sous-dossier dans le run MLflow (défaut: summaries/)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"{filename}_",
        delete=False, encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=2, default=str)
        tmp_path = Path(f.name)
    try:
        mlflow.log_artifact(str(tmp_path), artifact_path=artifact_path)
    except Exception as e:
        log.warning(f"[MLflow] Artifact JSON non loggué ({filename}) : {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Model Registry ─────────────────────────────────────────────────────────────

def registry_model_name(ticker: str, horizon: str) -> str:
    """
    Construit le nom standardisé d'un modèle dans le MLflow Model Registry.

    Convention : AlphaOps-Prophet-<Horizon>-<TICKER>
    Exemples   : AlphaOps-Prophet-Daily-AAPL
                 AlphaOps-Prophet-Monthly-BTC-USD

    Args:
        ticker  : symbole boursier (ex: "AAPL", "BTC-USD")
        horizon : "daily" ou "monthly" (insensible à la casse)
    """
    return f"AlphaOps-Prophet-{horizon.capitalize()}-{ticker.upper()}"


def register_prophet_models(
    run_id: str,
    models_daily: Dict,
    models_monthly: Dict,
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    Enregistre les modèles Prophet dans le MLflow Model Registry.

    À appeler APRÈS la fermeture du run MLflow (les modèles doivent déjà
    être loggués dans ce run via mlflow.prophet.log_model).

    Crée une nouvelle version pour chaque (ticker, horizon) dans le registry.
    Pose des tags de traçabilité sur chaque version.

    Args:
        run_id        : ID du run MLflow contenant les modèles loggués
        models_daily  : dict ticker → Prophet daily (déjà loggués dans le run)
        models_monthly: dict ticker → Prophet monthly
        extra_tags    : tags additionnels à poser sur chaque version
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    registered: list = []
    failed:     list = []

    for horizon, models in [("daily", models_daily), ("monthly", models_monthly)]:
        for ticker in models:
            model_name = registry_model_name(ticker, horizon)
            model_uri  = f"runs:/{run_id}/models/{horizon}/{ticker}"

            try:
                version = mlflow.register_model(model_uri=model_uri, name=model_name)

                # Tags de traçabilité sur la version enregistrée
                version_tags = {
                    "project":       "AlphaOps AI",
                    "model_family":  "prophet",
                    "horizon":       horizon,
                    "ticker":        ticker,
                    "source_run_id": run_id,
                    "trained_at":    datetime.now().strftime("%Y-%m-%d"),
                    **(extra_tags or {}),
                }
                for k, v in version_tags.items():
                    client.set_model_version_tag(model_name, version.version, k, v)

                registered.append(f"{model_name} v{version.version}")
                log.info(f"[MLflow Registry] {model_name} v{version.version} enregistré.")

            except Exception as e:
                failed.append(f"{model_name}: {e}")
                log.warning(f"[MLflow Registry] Échec {model_name} : {e}")

    if registered:
        print(f"  [MLflow Registry] {len(registered)} version(s) créée(s).")
        for name in registered:
            print(f"    [OK] {name}")
    if failed:
        print(f"  [MLflow Registry] {len(failed)} échec(s) :")
        for msg in failed:
            print(f"    [FAIL] {msg}")


def load_model_from_registry(
    model_name: str,
    stage_or_version: str = "latest",
) -> Any:
    """
    Charge un modèle Prophet depuis le MLflow Model Registry.

    Args:
        model_name       : nom complet du modèle dans le registry
                           (ex: "AlphaOps-Prophet-Daily-AAPL")
                           ou construit via registry_model_name(ticker, horizon)
        stage_or_version : "latest"  → dernière version disponible
                           "1", "2"  → numéro de version exact
                           "Staging" / "Production" → modèles promus

    Returns:
        Modèle Prophet chargé (objet Prophet prêt à appeler .predict())

    Exemples:
        from ml.registry.mlflow_utils import load_model_from_registry, registry_model_name

        # Dernière version AAPL daily
        model = load_model_from_registry(registry_model_name("AAPL", "daily"))

        # Version spécifique
        model = load_model_from_registry("AlphaOps-Prophet-Daily-AAPL", "2")

        # Modèle promu en production
        model = load_model_from_registry("AlphaOps-Prophet-Daily-AAPL", "Production")
    """
    import mlflow.prophet
    setup_mlflow()

    model_uri = f"models:/{model_name}/{stage_or_version}"
    log.info(f"[MLflow Registry] Chargement : {model_uri}")

    return mlflow.prophet.load_model(model_uri)
