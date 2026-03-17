import hashlib
import hmac
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AlphaOps Email Agent")

_mistral_key = os.getenv("MISTRAL_API_KEY")
if not _mistral_key:
    print("ERROR: MISTRAL_API_KEY is not set. Agent will fail on /notify requests.", flush=True)

llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=_mistral_key,
    temperature=0.4,
    max_tokens=1024,
)

HMAC_SECRET   = os.getenv("HMAC_SECRET", "alphaops-secret-change-me")
FEEDBACK_BASE = os.getenv("FEEDBACK_BASE_URL", "http://localhost:8082")

if HMAC_SECRET == "alphaops-secret-change-me":
    print("WARNING: HMAC_SECRET is set to the default value. Set a strong secret in production.", flush=True)


# ── Schemas ───────────────────────────────────────────────────────────────────

class LSTMSignals(BaseModel):
    prob_1d:   float = 0.5
    prob_7d:   float = 0.5
    prob_30d:  float = 0.5
    signal_1d:  str = "NEUTRE"
    signal_7d:  str = "NEUTRE"
    signal_30d: str = "NEUTRE"


class NotifyRequest(BaseModel):
    user_email:        str
    ticker:            str
    direction_daily:   bool | None = None   # Prophet J+1  (True = hausse)
    direction_monthly: bool | None = None   # Prophet J+30
    prob_daily:        float | None = None
    prob_month:        float | None = None
    lstm:              LSTMSignals | None = None


# ── HMAC helpers ──────────────────────────────────────────────────────────────

def make_token(ticker: str, pred_date: str, prediction: bool) -> str:
    msg = f"{ticker.upper()}:{pred_date}:{prediction}"
    return hmac.new(HMAC_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _dir_str(d: bool | None) -> str:
    if d is None:
        return "NEUTRE"
    return "HAUSSE ▲" if d else "BAISSE ▼"


def _prob_str(p: float | None) -> str:
    if p is None:
        return "N/A"
    return f"{p * 100:.1f}%"


# ── LangChain prompt ──────────────────────────────────────────────────────────

_EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un analyste financier expert, concis et professionnel. "
     "Tu rédiges des emails HTML pour des investisseurs particuliers en français. "
     "Retourne UNIQUEMENT le contenu HTML du corps (pas de balises html/head/body). "
      "Pas de formatage en markdown, ou blocs de code ou de html en markdown. "
     "Utilise uniquement du CSS inline. Sois direct, factuel, sans jargon excessif."),
    ("human",
     """Ticker analysé : {ticker}
Date d'analyse : {date}

=== Signaux Prophet (modèle de tendance) ===
Tendance J+1  : {dir_daily}   (probabilité hausse : {prob_daily})
Tendance J+30 : {dir_month}   (probabilité hausse : {prob_month})

=== Signaux LSTM (réseau de neurones) ===
J+1  : {signal_1d}  — prob. hausse : {prob_1d}
J+7  : {signal_7d}  — prob. hausse : {prob_7d}
J+30 : {signal_30d} — prob. hausse : {prob_30d}

Génère un email structuré avec exactement ces sections (titres en gras en html et pas en markdown) :
Résumé (2–3 phrases) — synthèse des signaux sur {ticker} sans jargon technique, et contexte général du marché. Par exemple, si les signaux sont mitigés, explique que les modèles sont incertains et que le marché est volatil, sans entrer dans les détails techniques de Prophet ou LSTM.
Interprétation — interprétation des signaux, points de vigilance, et contexte général du marché sans jargon technique. Ne parle ni de Prophet ni de LSTM, mais plutôt de "modèle de tendance" et "modèle de séquences temporelles". Sois factuel et évite les spéculations.
Recommandation — INVESTIR ou PATIENTER dans {ticker}, avec une justification en 1–2 phrases basée sur les signaux. Par exemple, si les signaux sont majoritairement positifs, recommander d'investir en expliquant que les modèles anticipent une hausse et que ça pourrait être une opportunité intéressante. Si les signaux sont mitigés ou négatifs, recommander de patienter en expliquant que les modèles sont incertains et que le marché est volatil.
Votre avis — demander si l'utilisateur valide ou conteste cette analyse (1-2 phrases, sans boutons).
"""),
])


def _generate_body(req: NotifyRequest, agree_url: str, disagree_url: str, pred_date: str) -> str:
    lstm = req.lstm or LSTMSignals()
    chain = _EMAIL_PROMPT | llm | StrOutputParser()
    llm_content = chain.invoke({
        "ticker":      req.ticker,
        "date":        pred_date,
        "dir_daily":   _dir_str(req.direction_daily),
        "prob_daily":  _prob_str(req.prob_daily),
        "dir_month":   _dir_str(req.direction_monthly),
        "prob_month":  _prob_str(req.prob_month),
        "signal_1d":   lstm.signal_1d,
        "prob_1d":     _prob_str(lstm.prob_1d),
        "signal_7d":   lstm.signal_7d,
        "prob_7d":     _prob_str(lstm.prob_7d),
        "signal_30d":  lstm.signal_30d,
        "prob_30d":    _prob_str(lstm.prob_30d),
    })
    buttons = f"""
<div style="margin-top:24px;text-align:center">
  <a href="{agree_url}"
     style="display:inline-block;padding:12px 28px;margin:6px;border-radius:6px;
            background:#238636;color:#ffffff;font-weight:700;text-decoration:none;font-size:14px">
    ✅ Bonne analyse - Je valide
  </a>
  <a href="{disagree_url}"
     style="display:inline-block;padding:12px 28px;margin:6px;border-radius:6px;
            background:#da3633;color:#ffffff;font-weight:700;text-decoration:none;font-size:14px">
    ❌ Analyse incorrecte
  </a>
</div>"""
    return llm_content + buttons


# ── HTML email template ───────────────────────────────────────────────────────

def _wrap_html(ticker: str, pred_date: str, content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AlphaOps - {ticker}</title>
</head>
<body style="margin:0;padding:0;font-family:Arial,sans-serif;color:#111111">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr><td align="center" style="padding:32px 16px">
      <table width="600" cellpadding="0" cellspacing="0"
             style="border:1px solid #dddddd;border-radius:8px;max-width:600px">

        <!-- Header -->
        <tr>
          <td style="padding:24px 32px;border-bottom:2px solid #0a84ff;border-radius:8px 8px 0 0">
            <span style="font-size:20px;font-weight:700;color:#0a84ff">📈 AlphaOps AI</span>
            <span style="float:right;font-size:12px;color:#666666;padding-top:5px">{pred_date}</span>
          </td>
        </tr>

        <!-- Ticker badge -->
        <tr>
          <td style="padding:20px 32px 0">
            <span style="border:1px solid #0a84ff;border-radius:4px;
                         padding:5px 12px;font-size:13px;color:#0a84ff;font-weight:700">
              {ticker}
            </span>
          </td>
        </tr>

        <!-- Body (LLM-generated) -->
        <tr>
          <td style="padding:20px 32px 32px;line-height:1.8;font-size:14px;color:#111111">
            {content}
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:16px 32px;border-top:1px solid #dddddd;text-align:center">
            <span style="font-size:11px;color:#999999">
              AlphaOps AI &middot; Analyse automatique &middot;
              Ne constitue pas un conseil financier réglementé.
            </span>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ── SMTP sender ───────────────────────────────────────────────────────────────

def _send_email(to: str, subject: str, html_body: str) -> None:
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_user or not smtp_pass:
        raise ValueError("SMTP_USER and SMTP_PASS must be set to send emails.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_from
    msg["To"]      = to
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as s:
        s.ehlo()
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)


# endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/notify")
def notify(req: NotifyRequest):
    ticker     = req.ticker.upper()
    pred_date  = str(date.today())
    prediction = req.direction_daily if req.direction_daily is not None else False
    token      = make_token(ticker, pred_date, prediction)

    pred_flag = "true" if prediction else "false"
    agree_url = (
        f"{FEEDBACK_BASE}/feedback/confirm"
        f"?token={token}&ticker={ticker}&date={pred_date}"
        f"&agreed=true&prediction={pred_flag}"
    )
    disagree_url = (
        f"{FEEDBACK_BASE}/feedback/confirm"
        f"?token={token}&ticker={ticker}&date={pred_date}"
        f"&agreed=false&prediction={pred_flag}"
    )

    content   = _generate_body(req, agree_url, disagree_url, pred_date)
    html_body = _wrap_html(ticker, pred_date, content)

    _send_email(
        to=req.user_email,
        subject=f"AlphaOps — Analyse {ticker} du {pred_date}",
        html_body=html_body,
    )

    return {"status": "sent", "ticker": ticker, "date": pred_date}
