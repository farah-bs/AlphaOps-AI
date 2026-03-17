#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# AlphaOps AI — Script de setup initial du VPS
#
# À exécuter UNE SEULE FOIS en root sur un VPS Ubuntu 22.04 / Debian 12 vierge.
# Usage :
#   curl -fsSL https://raw.githubusercontent.com/VOTRE_ORG/VOTRE_REPO/main/scripts/setup_vps.sh | bash
# Ou directement :
#   chmod +x setup_vps.sh && sudo bash setup_vps.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Variables à personnaliser ─────────────────────────────────────────────────
DEPLOY_USER="deploy"
APP_DIR="/opt/alphaops"
GIT_REPO="git@github.com:VOTRE_ORG/VOTRE_REPO.git"   # ← modifier
DOMAIN="VOTRE_DOMAINE"                                 # ← modifier
CERTBOT_EMAIL="VOTRE_EMAIL"                            # ← modifier
SSH_PORT=22                                            # Port SSH (22 par défaut)

# ── Couleurs ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✔]${NC} $*"; }
warn() { echo -e "${YELLOW}[⚠]${NC} $*"; }
err()  { echo -e "${RED}[✘]${NC} $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && err "Ce script doit être exécuté en root"

echo ""
echo "════════════════════════════════════════════"
echo "  AlphaOps AI — Setup VPS"
echo "════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. Mise à jour du système
# ══════════════════════════════════════════════════════════════════════════════
log "Mise à jour du système..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq
apt-get install -y -qq curl wget git unzip htop fail2ban ufw apache2-utils

# ══════════════════════════════════════════════════════════════════════════════
# 2. Utilisateur de déploiement (sans mot de passe sudo pour docker uniquement)
# ══════════════════════════════════════════════════════════════════════════════
log "Création de l'utilisateur '$DEPLOY_USER'..."
if ! id "$DEPLOY_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$DEPLOY_USER"
fi
# Répertoire SSH
mkdir -p /home/$DEPLOY_USER/.ssh
chmod 700 /home/$DEPLOY_USER/.ssh
chown -R $DEPLOY_USER:$DEPLOY_USER /home/$DEPLOY_USER/.ssh

warn "Ajoutez la clé publique SSH du runner CI dans /home/$DEPLOY_USER/.ssh/authorized_keys"
warn "  → Contenu du secret GitHub VPS_SSH_KEY (clé PUBLIQUE ici, privée dans le secret)"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Installation Docker + Docker Compose
# ══════════════════════════════════════════════════════════════════════════════
log "Installation de Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | bash
    systemctl enable --now docker
fi

usermod -aG docker "$DEPLOY_USER"
log "Docker $(docker --version) installé"

# ══════════════════════════════════════════════════════════════════════════════
# 4. Répertoire de l'application
# ══════════════════════════════════════════════════════════════════════════════
log "Création du répertoire applicatif $APP_DIR..."
mkdir -p "$APP_DIR"
chown -R $DEPLOY_USER:$DEPLOY_USER "$APP_DIR"

# Clone initial (la clé SSH deploy doit être configurée)
warn "Clonez le dépôt manuellement après avoir configuré la clé SSH :"
warn "  su - $DEPLOY_USER"
warn "  git clone $GIT_REPO $APP_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# 5. Firewall UFW
# ══════════════════════════════════════════════════════════════════════════════
log "Configuration du firewall UFW..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow $SSH_PORT/tcp   comment "SSH"
ufw allow 80/tcp          comment "HTTP (Let's Encrypt + redirect)"
ufw allow 443/tcp         comment "HTTPS"
ufw --force enable
log "UFW activé. Ports ouverts : $SSH_PORT, 80, 443"

# ── Note importante : Docker bypass UFW via iptables ─────────────────────────
# Pour éviter que Docker n'expose des ports directement (bypass UFW),
# tous les ports sensibles sont bindés sur 127.0.0.1 dans docker-compose.prod.yml
log "Note : ports Docker bindés sur 127.0.0.1 dans docker-compose.prod.yml → UFW respecté"

# ══════════════════════════════════════════════════════════════════════════════
# 6. Fail2ban (protection SSH brute-force)
# ══════════════════════════════════════════════════════════════════════════════
log "Configuration de Fail2ban..."
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime  = 3600
findtime = 600
maxretry = 5
backend  = systemd

[sshd]
enabled  = true
port     = $SSH_PORT
logpath  = %(sshd_log)s

[nginx-limit-req]
enabled  = true
port     = http,https
logpath  = /var/log/nginx/error.log
maxretry = 20
EOF

systemctl enable --now fail2ban
log "Fail2ban actif (ban après 5 tentatives en 10min, durée 1h)"

# ══════════════════════════════════════════════════════════════════════════════
# 7. Hardening SSH
# ══════════════════════════════════════════════════════════════════════════════
log "Durcissement SSH..."
SSHD_CONFIG="/etc/ssh/sshd_config"
# Sauvegarde
cp "$SSHD_CONFIG" "${SSHD_CONFIG}.bak"

# Appliquer les paramètres sécurisés
sed -i "s/^#*PermitRootLogin.*/PermitRootLogin no/"          "$SSHD_CONFIG"
sed -i "s/^#*PasswordAuthentication.*/PasswordAuthentication no/" "$SSHD_CONFIG"
sed -i "s/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/"   "$SSHD_CONFIG"
sed -i "s/^#*MaxAuthTries.*/MaxAuthTries 3/"                    "$SSHD_CONFIG"
sed -i "s/^#*LoginGraceTime.*/LoginGraceTime 30/"               "$SSHD_CONFIG"
sed -i "s/^#*X11Forwarding.*/X11Forwarding no/"                 "$SSHD_CONFIG"

systemctl reload sshd
log "SSH : connexion root désactivée, authentification par clé uniquement"

# ══════════════════════════════════════════════════════════════════════════════
# 8. Certificat SSL (Let's Encrypt via certbot standalone)
# ══════════════════════════════════════════════════════════════════════════════
log "Génération du certificat SSL pour $DOMAIN..."
# Certbot tourne dans Docker (voir docker-compose.prod.yml)
# Ce bloc génère le certificat initial AVANT de démarrer Nginx

if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    apt-get install -y -qq certbot
    # Port 80 doit être libre à ce stade (Nginx pas encore démarré)
    certbot certonly \
        --standalone \
        --non-interactive \
        --agree-tos \
        --email "$CERTBOT_EMAIL" \
        -d "$DOMAIN" \
        --rsa-key-size 4096

    # Copier vers les volumes Docker attendus par nginx
    mkdir -p "$APP_DIR/nginx/ssl"
    log "Certificat SSL généré pour $DOMAIN"
else
    log "Certificat SSL déjà présent pour $DOMAIN"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 9. Fichier .htpasswd pour Airflow et MLflow
# ══════════════════════════════════════════════════════════════════════════════
log "Génération du fichier .htpasswd..."
mkdir -p "$APP_DIR/nginx"
HTPASSWD_FILE="$APP_DIR/nginx/.htpasswd"

warn "Entrez le mot de passe pour l'accès Airflow/MLflow :"
read -r -s -p "Mot de passe : " HTPASSWD_PWD
echo ""
htpasswd -cb "$HTPASSWD_FILE" admin "$HTPASSWD_PWD"
chmod 640 "$HTPASSWD_FILE"
chown root:$DEPLOY_USER "$HTPASSWD_FILE"

log ".htpasswd créé (utilisateur : admin)"
warn "Ajoutez le contenu de $HTPASSWD_FILE dans le secret GitHub HTPASSWD"

# ══════════════════════════════════════════════════════════════════════════════
# 10. Cron de renouvellement SSL
# ══════════════════════════════════════════════════════════════════════════════
log "Configuration du cron de renouvellement SSL..."
(crontab -l 2>/dev/null; echo "0 3 * * * docker compose -f $APP_DIR/docker-compose.yml -f $APP_DIR/docker-compose.prod.yml exec certbot certbot renew --quiet && docker compose -f $APP_DIR/docker-compose.yml -f $APP_DIR/docker-compose.prod.yml exec nginx nginx -s reload") | crontab -
log "Cron SSL : renouvellement automatique à 3h chaque jour"

# ══════════════════════════════════════════════════════════════════════════════
# 11. Résumé final
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Setup VPS terminé !"
echo "════════════════════════════════════════════"
echo ""
echo "  Prochaines étapes manuelles :"
echo ""
echo "  1. Ajouter la clé publique SSH dans /home/$DEPLOY_USER/.ssh/authorized_keys"
echo ""
echo "  2. Cloner le dépôt :"
echo "     su - $DEPLOY_USER"
echo "     git clone $GIT_REPO $APP_DIR"
echo ""
echo "  3. Créer le fichier .env de production dans $APP_DIR/.env"
echo "     (copier .env.example et remplir les valeurs)"
echo ""
echo "  4. Ajouter les secrets dans GitHub (Settings → Secrets → Actions) :"
echo "     VPS_HOST    → $DOMAIN (ou IP)"
echo "     VPS_USER    → $DEPLOY_USER"
echo "     VPS_SSH_KEY → Clé privée SSH du runner"
echo "     VPS_PORT    → $SSH_PORT"
echo "     ENV_PROD    → Contenu du fichier .env"
echo "     HTPASSWD    → Contenu de $HTPASSWD_FILE"
echo ""
echo "  5. Premier démarrage :"
echo "     cd $APP_DIR"
echo "     docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
echo ""
echo "  6. Vérifier les services :"
echo "     docker compose -f docker-compose.yml -f docker-compose.prod.yml ps"
echo ""
