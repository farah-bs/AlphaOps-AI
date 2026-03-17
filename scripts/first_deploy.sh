#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# AlphaOps AI — Premier démarrage sur le VPS (après setup_vps.sh)
#
# À exécuter EN TANT QUE deploy depuis /opt/alphaops
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

APP_DIR="/opt/alphaops"
cd "$APP_DIR"

echo "[1/4] Création des répertoires nécessaires..."
mkdir -p artifacts mlruns airflow/logs airflow/plugins nginx

echo "[2/4] Vérification du .env..."
if [ ! -f ".env" ]; then
    echo "❌ Fichier .env manquant. Créez-le depuis .env.example"
    exit 1
fi

echo "[3/4] Build et démarrage des services..."
docker compose \
    -f docker-compose.yml \
    -f docker-compose.prod.yml \
    up -d --build

echo "[4/4] État des services :"
docker compose \
    -f docker-compose.yml \
    -f docker-compose.prod.yml \
    ps

echo ""
echo "✅ AlphaOps AI démarré !"
echo "   → https://VOTRE_DOMAINE"
echo "   → https://VOTRE_DOMAINE/airflow  (login requis)"
echo "   → https://VOTRE_DOMAINE/mlflow   (login requis)"
