import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.agents.agent import ask_agent
from src.db.connection import get_engine

# Chemin vers les artifacts de backtest (robuste local + Docker)
# webapp/app.py → parent = webapp/ → parent.parent = racine projet → artifacts/
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Rend ml.* importable :
#   Docker  : app.py est /app/app.py  → /app contient /app/ml/
#   Local   : app.py est .../webapp/app.py → parent.parent = racine projet
_APP_DIR = Path(__file__).resolve().parent
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SERVING_URL = "http://serving:8080"
AGENT_URL   = "http://agent:8083"

st.set_page_config(page_title="AlphaOps AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.5); font-size: 0.75rem; }
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 AlphaOps AI")

# ── Helpers DB ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_history(ticker: str, days: int = 90) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT date, open_price, high_price, low_price, adj_close, volume
        FROM fact_ohlcv
        WHERE symbol = :symbol
        ORDER BY date DESC
        LIMIT :days
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": ticker, "days": days})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Chart 2D ──────────────────────────────────────────────────────────────────

def make_2d_chart(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    ticker: str,
    direction: bool,
) -> go.Figure:
    clr_hist  = "#00d4ff"
    clr_pred  = "#ff6b35"
    title_clr = "#00ff88" if direction else "#ff4466"
    arrow     = "▲" if direction else "▼"
    label     = "HAUSSE" if direction else "BAISSE"

    fig = go.Figure()

    # Fourchette historique H-L (zone)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_hist["date"], df_hist["date"].iloc[::-1]]),
        y=pd.concat([df_hist["high_price"], df_hist["low_price"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Fourchette H/L",
        hoverinfo="skip",
    ))

    # Ligne des prix réels
    fig.add_trace(go.Scatter(
        x=df_hist["date"],
        y=df_hist["adj_close"],
        name="Prix réel",
        line=dict(color=clr_hist, width=2),
        mode="lines",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,.2f} $<extra>Historique</extra>",
    ))

    # Zone d'incertitude forecast
    dates_fwd = df_pred["date"].tolist()
    fig.add_trace(go.Scatter(
        x=dates_fwd + dates_fwd[::-1],
        y=df_pred["yhat_upper"].tolist() + df_pred["yhat_lower"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(255, 107, 53, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Incertitude",
        hoverinfo="skip",
    ))

    # Ligne forecast
    fig.add_trace(go.Scatter(
        x=df_pred["date"],
        y=df_pred["yhat"],
        name="Prévision",
        line=dict(color=clr_pred, width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6, color=clr_pred, symbol="circle"),
        customdata=np.stack([df_pred["yhat_lower"], df_pred["yhat_upper"]], axis=-1),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Prévision : <b>%{y:,.2f} $</b><br>"
            "↓ %{customdata[0]:,.2f}  |  ↑ %{customdata[1]:,.2f}"
            "<extra>Forecast</extra>"
        ),
    ))

    # Ligne verticale aujourd'hui
    today_str = pd.Timestamp.today().normalize().isoformat()
    fig.add_shape(
        type="line",
        x0=today_str, x1=today_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1.5),
    )
    fig.add_annotation(
        x=today_str, y=1, yref="paper",
        text="  Aujourd'hui",
        showarrow=False,
        font=dict(color="rgba(255,255,255,0.35)", size=11),
        xanchor="left",
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b>  <span style='color:{title_clr}'>{arrow} {label}</span>",
            font=dict(size=20),
            x=0.01,
        ),
        xaxis=dict(
            title="Date",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Prix (USD)",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=440,
        plot_bgcolor="rgba(8,12,24,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=20, t=80, b=50),
    )
    return fig


# ── Chart 3D LSTM ─────────────────────────────────────────────────────────────

def _make_lstm_3d_chart(
    ticker: str,
    prob_1d: float,
    prob_7d: float,
    prob_30d: float,
    df_hist: pd.DataFrame,
) -> go.Figure:
    """
    Surface 3D combinée :
      - Section gauche  : tunnel historique prix (low/close/high)  → dégradé cyan
      - Section droite  : cône probabiliste LSTM (J+1 / J+7 / J+30)
        → couleur cyan (HAUSSE) ou orange (BAISSE) selon le signal moyen
    Le tout connecté par un plan «Aujourd'hui» semi-transparent.
    """
    fig = go.Figure()

    n_hist = len(df_hist)
    y_lvl  = np.array([0.0, 0.5, 1.0])
    x_hist = np.arange(n_hist, dtype=float)
    last_close = float(df_hist["adj_close"].iloc[-1])

    # Volatilité quotidienne historique (pour l'incertitude forward)
    returns   = df_hist["adj_close"].pct_change().dropna()
    daily_vol = float(returns.std()) if len(returns) > 1 else 0.01

    # ── Surface historique ──────────────────────────────────────────────────
    z_hist = np.array([
        df_hist["low_price"].values,
        df_hist["adj_close"].values,
        df_hist["high_price"].values,
    ])
    fig.add_trace(go.Surface(
        x=x_hist, y=y_lvl, z=z_hist,
        colorscale=[
            [0.0, "#050d1f"], [0.25, "#0b2a52"],
            [0.6,  "#005f99"], [1.0,  "#00d4ff"],
        ],
        opacity=0.75,
        showscale=False,
        name="Historique",
        hovertemplate="T=%{x:.0f}<br>Prix=%{z:,.2f} $<extra>Historique</extra>",
    ))
    # Spine historique
    fig.add_trace(go.Scatter3d(
        x=x_hist, y=np.full(n_hist, 0.5), z=df_hist["adj_close"].values,
        mode="lines",
        line=dict(color="#00d4ff", width=5),
        name="Clôture réelle",
    ))

    # ── Cône LSTM (scénario probabiliste) ──────────────────────────────────
    # Points : [aujourd'hui=0, J+1, J+7, J+30]
    horizons_days = np.array([0,    1,       7,       30])
    probs_all     = np.array([0.5,  prob_1d, prob_7d, prob_30d])

    # Drift de prix attendu = (prob - 0.5) * 2 * daily_vol par jour (log-drift)
    drifts  = (probs_all - 0.5) * 2 * daily_vol
    centers = last_close * np.exp(drifts * horizons_days)

    # Enveloppe d'incertitude croissante (σ√t)
    unc = last_close * daily_vol * np.sqrt(np.maximum(horizons_days, 0.1))

    x_fwd = (n_hist - 1) + horizons_days.astype(float)
    z_fwd = np.array([centers - unc, centers, centers + unc])

    avg_prob = (prob_1d + prob_7d + prob_30d) / 3
    if avg_prob > 0.55:
        lstm_cs = [[0.0, "#002a1a"], [0.4, "#006644"], [0.7, "#00bb77"], [1.0, "#00d4ff"]]
        spine_color = "#00d4ff"
    elif avg_prob < 0.45:
        lstm_cs = [[0.0, "#1a0600"], [0.4, "#6b2000"], [0.7, "#cc4a00"], [1.0, "#ff6b35"]]
        spine_color = "#ff6b35"
    else:
        lstm_cs = [[0.0, "#111111"], [0.5, "#555555"], [1.0, "#aaaaaa"]]
        spine_color = "#aaaaaa"

    fig.add_trace(go.Surface(
        x=x_fwd, y=y_lvl, z=z_fwd,
        colorscale=lstm_cs,
        opacity=0.88,
        showscale=False,
        name="Cône LSTM",
        hovertemplate="T=%{x:.0f}<br>%{z:,.2f} $<extra>Scénario LSTM</extra>",
    ))
    # Spine LSTM (médian)
    fig.add_trace(go.Scatter3d(
        x=x_fwd, y=np.full(len(x_fwd), 0.5), z=centers,
        mode="lines",
        line=dict(color=spine_color, width=6),
        name="Forecast médian",
    ))
    # Points clés J+1, J+7, J+30 avec labels
    key_x    = [n_hist + d - 1 for d in [1, 7, 30]]
    key_z    = [centers[1], centers[2], centers[3]]
    key_cols = [_gauge_color(prob_1d), _gauge_color(prob_7d), _gauge_color(prob_30d)]
    key_lbls = [
        f"J+1 {prob_1d*100:.0f}%",
        f"J+7 {prob_7d*100:.0f}%",
        f"J+30 {prob_30d*100:.0f}%",
    ]
    fig.add_trace(go.Scatter3d(
        x=key_x, y=[0.5, 0.5, 0.5], z=key_z,
        mode="markers+text",
        marker=dict(size=11, color=key_cols, symbol="diamond",
                    line=dict(color="white", width=1)),
        text=key_lbls,
        textposition="top center",
        textfont=dict(color="white", size=11),
        name="Points LSTM",
    ))

    # ── Plan de séparation «Aujourd'hui» ────────────────────────────────────
    p_min = float(df_hist["low_price"].min()) * 0.97
    p_max = float(df_hist["high_price"].max()) * 1.03
    sep_x = float(n_hist - 1)
    fig.add_trace(go.Mesh3d(
        x=[sep_x, sep_x, sep_x, sep_x],
        y=[0.0, 1.0, 1.0, 0.0],
        z=[p_min, p_min, p_max, p_max],
        i=[0], j=[1], k=[2],
        color="white", opacity=0.07,
        showlegend=False, hoverinfo="skip",
    ))

    sig_label = "HAUSSE ▲" if avg_prob > 0.55 else ("BAISSE ▼" if avg_prob < 0.45 else "NEUTRE —")
    sig_color  = "#00d4ff" if avg_prob > 0.55 else ("#ff6b35" if avg_prob < 0.45 else "#888888")

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                f"<b>{ticker}</b> — Tunnel de prix + Cône LSTM  "
                f"<span style='color:{sig_color}'>{sig_label}</span>"
            ),
            font=dict(size=17), x=0.01,
        ),
        scene=dict(
            xaxis=dict(title="Temps (index)",
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       backgroundcolor="rgba(5,8,18,0.6)", showspikes=False),
            yaxis=dict(title="Niveau canal",
                       tickvals=[0.0, 0.5, 1.0], ticktext=["Bas", "Médian", "Haut"],
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       backgroundcolor="rgba(5,8,18,0.6)", showspikes=False),
            zaxis=dict(title="Prix (USD)",
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       backgroundcolor="rgba(5,8,18,0.6)", showspikes=False),
            camera=dict(eye=dict(x=1.6, y=-1.9, z=0.85)),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.7, z=1.0),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=600,
        margin=dict(l=0, r=0, t=65, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.4)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1, font=dict(size=12)),
    )
    return fig


# ── Chart 3D ──────────────────────────────────────────────────────────────────

def make_3d_chart(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Surface 3D "tunnel de prix" :
      X = index temporel
      Y = niveau canal  (0 = bas, 0.5 = médian, 1 = haut)
      Z = prix (USD)    ← la hauteur de la surface

    Section historique  → low / adj_close / high       (bleu)
    Section forecast    → yhat_lower / yhat / yhat_upper (orange)
    """
    fig = go.Figure()

    n_hist = len(df_hist)
    n_pred = len(df_pred)
    x_hist = np.arange(n_hist, dtype=float)
    x_pred = np.arange(n_hist, n_hist + n_pred, dtype=float)
    y_lvl  = np.array([0.0, 0.5, 1.0])

    # ── Surface historique ──────────────────────────────────────────────────
    z_hist = np.array([
        df_hist["low_price"].values,
        df_hist["adj_close"].values,
        df_hist["high_price"].values,
    ])  # shape (3, n_hist)

    fig.add_trace(go.Surface(
        x=x_hist,
        y=y_lvl,
        z=z_hist,
        colorscale=[
            [0.0,  "#050d1f"],
            [0.25, "#0b2a52"],
            [0.6,  "#005f99"],
            [1.0,  "#00d4ff"],
        ],
        opacity=0.82,
        showscale=False,
        name="Historique",
        hovertemplate="T=%{x:.0f}<br>Prix=%{z:,.2f} $<extra>Historique</extra>",
    ))

    # Ligne spine historique (clôture réelle)
    fig.add_trace(go.Scatter3d(
        x=x_hist,
        y=np.full(n_hist, 0.5),
        z=df_hist["adj_close"].values,
        mode="lines",
        line=dict(color="#00d4ff", width=5),
        name="Clôture réelle",
    ))

    # ── Surface forecast ────────────────────────────────────────────────────
    z_pred = np.array([
        df_pred["yhat_lower"].values,
        df_pred["yhat"].values,
        df_pred["yhat_upper"].values,
    ])  # shape (3, n_pred)

    fig.add_trace(go.Surface(
        x=x_pred,
        y=y_lvl,
        z=z_pred,
        colorscale=[
            [0.0,  "#1a0600"],
            [0.25, "#6b2000"],
            [0.6,  "#cc4a00"],
            [1.0,  "#ff6b35"],
        ],
        opacity=0.85,
        showscale=False,
        name="Prévision",
        hovertemplate="T=%{x:.0f}<br>Prix=%{z:,.2f} $<extra>Forecast</extra>",
    ))

    # Ligne spine forecast (yhat)
    fig.add_trace(go.Scatter3d(
        x=x_pred,
        y=np.full(n_pred, 0.5),
        z=df_pred["yhat"].values,
        mode="lines+markers",
        line=dict(color="#ff6b35", width=5),
        marker=dict(size=4, color="#ff6b35"),
        name="Forecast médian",
    ))

    # ── Plan de séparation "aujourd'hui" ────────────────────────────────────
    sep_x = float(n_hist - 1)
    p_min = min(float(df_hist["low_price"].min()), float(df_pred["yhat_lower"].min()))
    p_max = max(float(df_hist["high_price"].max()), float(df_pred["yhat_upper"].max()))

    fig.add_trace(go.Mesh3d(
        x=[sep_x, sep_x, sep_x, sep_x],
        y=[0.0, 1.0, 1.0, 0.0],
        z=[p_min, p_min, p_max, p_max],
        i=[0], j=[1], k=[2],
        color="white",
        opacity=0.08,
        name="Aujourd'hui",
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — Tunnel de prix 3D",
            font=dict(size=17),
            x=0.01,
        ),
        scene=dict(
            xaxis=dict(
                title="Temps (index)",
                showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                backgroundcolor="rgba(5,8,18,0.6)",
                showspikes=False,
            ),
            yaxis=dict(
                title="Niveau canal",
                tickvals=[0.0, 0.5, 1.0],
                ticktext=["Bas", "Médian", "Haut"],
                showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                backgroundcolor="rgba(5,8,18,0.6)",
                showspikes=False,
            ),
            zaxis=dict(
                title="Prix (USD)",
                showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                backgroundcolor="rgba(5,8,18,0.6)",
                showspikes=False,
            ),
            camera=dict(eye=dict(x=1.6, y=-1.9, z=0.85)),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.7, z=1.0),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=580,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    return fig


# ── Helpers gauges et couleurs ────────────────────────────────────────────────

def _gauge_color(prob: float) -> str:
    """Couleur selon la probabilité : cyan (hausse), orange (baisse), gris (neutre)."""
    if prob > 0.55:
        return "#00d4ff"
    if prob < 0.45:
        return "#ff6b35"
    return "#888888"


def _make_gauge(label: str, prob: float) -> go.Figure:
    """Gauge Plotly Indicator pour une proba directionnelle."""
    color = _gauge_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        title={"text": label, "font": {"size": 14, "color": "rgba(255,255,255,0.7)"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  45], "color": "rgba(255,107,53,0.15)"},
                {"range": [45, 55], "color": "rgba(136,136,136,0.1)"},
                {"range": [55, 100],"color": "rgba(0,212,255,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=10, l=20, r=20),
        height=220,
    )
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_predict, tab_backtest, tab_conseil = st.tabs(
    ["💬 Chatbot", "🔮 Prédiction", "📊 Backtest", "💡 Conseil"]
)

# ── Onglet Chatbot ────────────────────────────────────────────────────────────
with tab_chat:
    with st.sidebar:
        st.markdown("### Options")
        show_debug = st.toggle("Afficher debug (SQL/Validation/Raw)", value=True)
        if st.button("🧹 Reset conversation"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_prompt = st.chat_input("Pose une question (ex: 'Prix NVDA hier ?')")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Je récupère les données et je génère la requête SQL..."):
                try:
                    res = ask_agent(user_prompt)
                    answer = res.get("answer", "Je n'ai pas de réponse.")
                    st.markdown(answer)

                    if show_debug:
                        with st.expander("🧾 SQL généré"):
                            st.code(res.get("sql", ""), language="sql")
                        with st.expander("✅ Validation"):
                            st.json(res.get("validation", {}))
                        with st.expander("📦 Résultat brut"):
                            st.write(res.get("raw_result"))

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    err = f"Erreur: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})


# ── Onglet Prédiction (LSTM) ───────────────────────────────────────────────────
with tab_predict:
    st.subheader("Prédiction de marché — LSTM")
    st.caption(
        "Le modèle LSTM prédit la **probabilité de hausse** à J+1, J+7 et J+30. "
        "Signal HAUSSE si prob > 55 %, BAISSE si < 45 %, NEUTRE sinon."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.selectbox(
            "Ticker",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD"],
            key="pred_ticker_select",
        )
    with col2:
        st.write("")
        st.write("")
        predict_btn = st.button("🔍 Analyser", use_container_width=True, type="primary", key="pred_lstm_btn")

    if predict_btn:
        with st.spinner(f"Prédiction LSTM pour {ticker}…"):
            try:
                resp = requests.post(
                    f"{SERVING_URL}/predict/lstm",
                    json={"ticker": ticker},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state["last_lstm_prediction"] = {
                    "ticker":     ticker,
                    "prob_1d":    data.get("prob_up_1d",  0.5),
                    "prob_7d":    data.get("prob_up_7d",  0.5),
                    "prob_30d":   data.get("prob_up_30d", 0.5),
                    "signal_1d":  data.get("signal_1d",  "—"),
                    "signal_7d":  data.get("signal_7d",  "—"),
                    "signal_30d": data.get("signal_30d", "—"),
                }
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de joindre le service serving (http://serving:8080).")
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    st.warning(f"⚠️ Aucun modèle LSTM trouvé pour **{ticker}**. Lancez d'abord le DAG `lstm_training` dans Airflow.")
                else:
                    st.error(f"❌ Erreur API : {e}")
            except Exception as e:
                st.error(f"❌ Erreur inattendue : {e}")

    if "last_lstm_prediction" in st.session_state:
        last = st.session_state["last_lstm_prediction"]
        tk        = last["ticker"]
        prob_1d   = last["prob_1d"]
        prob_7d   = last["prob_7d"]
        prob_30d  = last["prob_30d"]
        sig_1d    = last["signal_1d"]
        sig_7d    = last["signal_7d"]
        sig_30d   = last["signal_30d"]

        # ── Métriques ────────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Ticker", tk)
        with mc2:
            st.metric("Signal J+1",  sig_1d,  f"{prob_1d*100:.1f}%")
        with mc3:
            st.metric("Signal J+7",  sig_7d,  f"{prob_7d*100:.1f}%")
        with mc4:
            st.metric("Signal J+30", sig_30d, f"{prob_30d*100:.1f}%")

        st.divider()

        # ── Jauges LSTM ───────────────────────────────────────────────────────
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.plotly_chart(
                _make_gauge("Hausse J+1", prob_1d),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with gc2:
            st.plotly_chart(
                _make_gauge("Hausse J+7", prob_7d),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with gc3:
            st.plotly_chart(
                _make_gauge("Hausse J+30", prob_30d),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.caption(
            "🔵 Cyan = HAUSSE (prob > 55 %)  ·  "
            "🟠 Orange = BAISSE (prob < 45 %)  ·  "
            "⬜ Gris = NEUTRE"
        )

        st.divider()

        # ── Graphique barres des probabilités par horizon ─────────────────────
        horizons   = ["J+1", "J+7", "J+30"]
        probs_pct  = [prob_1d * 100, prob_7d * 100, prob_30d * 100]
        bar_colors = [_gauge_color(p / 100) for p in probs_pct]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=horizons,
            y=probs_pct,
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in probs_pct],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Prob. hausse : %{y:.1f}%<extra></extra>",
        ))
        fig_bar.add_hline(
            y=55, line=dict(color="rgba(0,212,255,0.5)", dash="dot", width=1.5),
            annotation_text="Seuil HAUSSE (55%)",
            annotation_font_color="rgba(0,212,255,0.7)",
            annotation_position="top left",
        )
        fig_bar.add_hline(
            y=45, line=dict(color="rgba(255,107,53,0.5)", dash="dot", width=1.5),
            annotation_text="Seuil BAISSE (45%)",
            annotation_font_color="rgba(255,107,53,0.7)",
            annotation_position="bottom left",
        )
        fig_bar.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{tk}</b> — Probabilité de hausse LSTM par horizon",
                font=dict(size=17), x=0.01,
            ),
            xaxis=dict(title="Horizon"),
            yaxis=dict(title="Prob. hausse (%)", range=[0, 105]),
            height=340,
            plot_bgcolor="rgba(8,12,24,0.9)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=55, r=20, t=70, b=50),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ── Historique 2D avec annotation signal ──────────────────────────────
        df_hist_lstm = pd.DataFrame()
        try:
            df_hist_lstm = fetch_history(tk, days=90)
            if not df_hist_lstm.empty:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=pd.concat([df_hist_lstm["date"], df_hist_lstm["date"].iloc[::-1]]),
                    y=pd.concat([df_hist_lstm["high_price"], df_hist_lstm["low_price"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0, 212, 255, 0.07)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Fourchette H/L",
                    hoverinfo="skip",
                ))
                fig_hist.add_trace(go.Scatter(
                    x=df_hist_lstm["date"],
                    y=df_hist_lstm["adj_close"],
                    name="Prix réel",
                    line=dict(color="#00d4ff", width=2),
                    mode="lines",
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,.2f} $<extra>Historique</extra>",
                ))

                last_date  = df_hist_lstm["date"].iloc[-1]
                last_close = float(df_hist_lstm["adj_close"].iloc[-1])
                sig_color  = "#00d4ff" if sig_1d == "HAUSSE" else ("#ff6b35" if sig_1d == "BAISSE" else "#888888")
                arrow_sym  = "▲" if sig_1d == "HAUSSE" else ("▼" if sig_1d == "BAISSE" else "—")
                fig_hist.add_annotation(
                    x=last_date, y=last_close,
                    text=f"  {arrow_sym} {sig_1d} J+1",
                    showarrow=True, arrowhead=2, arrowcolor=sig_color,
                    font=dict(color=sig_color, size=13, family="monospace"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor=sig_color, borderwidth=1, borderpad=4,
                )
                fig_hist.update_layout(
                    template="plotly_dark",
                    title=dict(text=f"<b>{tk}</b> — Historique 90j + signal LSTM J+1",
                               font=dict(size=17), x=0.01),
                    xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Prix (USD)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400,
                    plot_bgcolor="rgba(8,12,24,0.9)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=55, r=20, t=70, b=50),
                )
                st.plotly_chart(fig_hist, use_container_width=True,
                                config={"displayModeBar": True, "scrollZoom": True})
        except Exception:
            pass

        # ── Vue 3D : Tunnel historique + Cône LSTM ───────────────────────────
        if not df_hist_lstm.empty:
            st.markdown("#### 🧊 Vue 3D — Tunnel de prix + Cône probabiliste LSTM")
            st.caption(
                "Surface bleue = historique (bas=low, médian=clôture, haut=high)  ·  "
                "Cône coloré = scénario LSTM (élargissement = incertitude √t)  ·  "
                "♦ = points J+1, J+7, J+30"
            )
            st.plotly_chart(
                _make_lstm_3d_chart(tk, prob_1d, prob_7d, prob_30d, df_hist_lstm),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )

        # ── Tableau récapitulatif ─────────────────────────────────────────────
        st.divider()
        summary_rows = [
            {"Horizon": "J+1",  "Prob. hausse": f"{prob_1d*100:.1f}%",  "Signal": sig_1d},
            {"Horizon": "J+7",  "Prob. hausse": f"{prob_7d*100:.1f}%",  "Signal": sig_7d},
            {"Horizon": "J+30", "Prob. hausse": f"{prob_30d*100:.1f}%", "Signal": sig_30d},
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    else:
        st.info("👆 Sélectionnez un ticker et cliquez sur **Analyser** pour obtenir les signaux LSTM.")


# ── Onglet Backtest ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_backtest_detail(ticker: str) -> pd.DataFrame:
    """Charge le CSV de détail du backtest depuis artifacts/."""
    path = ARTIFACTS_DIR / f"backtest_{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["target_date", "train_end"])
    return df


@st.cache_data(ttl=60)
def _load_backtest_summary(ticker: str) -> dict:
    """Charge le JSON de synthèse du backtest depuis artifacts/."""
    path = ARTIFACTS_DIR / f"backtest_summary_{ticker}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _make_backtest_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Graphique Actual vs Predicted (yhat) avec l'intervalle de confiance.
    Style cohérent avec les graphiques existants (plotly_dark).
    """
    fig = go.Figure()

    # Intervalle de confiance [yhat_lower, yhat_upper]
    fig.add_trace(go.Scatter(
        x=pd.concat([df["target_date"], df["target_date"].iloc[::-1]]),
        y=pd.concat([df["yhat_upper"], df["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(255, 107, 53, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalle confiance",
        hoverinfo="skip",
    ))

    # Prédiction (yhat)
    fig.add_trace(go.Scatter(
        x=df["target_date"],
        y=df["yhat"],
        name="Prédiction (yhat)",
        line=dict(color="#ff6b35", width=1.8, dash="dash"),
        mode="lines",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>yhat : %{y:,.2f} $<extra>Prédiction</extra>",
    ))

    # Prix réel
    fig.add_trace(go.Scatter(
        x=df["target_date"],
        y=df["actual"],
        name="Prix réel",
        line=dict(color="#00d4ff", width=2),
        mode="lines",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Réel : %{y:,.2f} $<extra>Réel</extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — Backtest : Réel vs Prédit",
            font=dict(size=18),
            x=0.01,
        ),
        xaxis=dict(
            title="Date",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Prix (USD)",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        plot_bgcolor="rgba(8,12,24,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=20, t=70, b=50),
    )
    return fig


def _make_error_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Graphique de l'erreur absolue dans le temps."""
    fig = go.Figure()

    # Zone de couverture (interval_hit = 1 en vert clair, 0 en rouge clair)
    colors = ["rgba(0,255,136,0.5)" if h else "rgba(255,68,102,0.5)"
              for h in df["interval_hit"]]

    fig.add_trace(go.Bar(
        x=df["target_date"],
        y=df["abs_error"],
        marker_color=colors,
        name="Erreur absolue",
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Erreur abs : %{y:,.2f} $<br>"
            "<extra></extra>"
        ),
    ))

    # Ligne de MAE moyen
    mae_val = float(df["abs_error"].mean())
    fig.add_hline(
        y=mae_val,
        line=dict(color="rgba(255,255,0,0.6)", dash="dot", width=1.5),
        annotation_text=f"MAE = {mae_val:.2f}",
        annotation_font_color="rgba(255,255,0,0.8)",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — Erreur absolue  "
                 "<span style='color:#00ff88'>■ dans l'intervalle</span>  "
                 "<span style='color:#ff4466'>■ hors intervalle</span>",
            font=dict(size=16),
            x=0.01,
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Erreur absolue ($)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        height=320,
        plot_bgcolor="rgba(8,12,24,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=20, t=60, b=50),
        showlegend=False,
    )
    return fig


# ── Constantes animation backtest ─────────────────────────────────────────────
_BT_SPINNERS = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
_BT_TERMINAL_CSS = """
<style>
@keyframes bt-glow {
  0%, 100% { border-color: #00d4ff55; box-shadow: 0 0 4px #00d4ff22; }
  50%       { border-color: #00d4ffaa; box-shadow: 0 0 14px #00d4ff44; }
}
.bt-terminal {
  background: #0d1117;
  border: 1px solid #00d4ff55;
  border-radius: 10px;
  padding: 16px 20px;
  font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.6;
  max-height: 280px;
  overflow-y: auto;
  animation: bt-glow 2s ease-in-out infinite;
}
</style>
"""


def _bt_terminal_html(spinner: str, pct: float, count: int, total: int, lines: list) -> str:
    """Génère le HTML du terminal animé pendant le backtest."""
    log_html = "<br>".join(
        f'<span style="color:#8b949e">{ln}</span>' for ln in lines[-14:]
    )
    dots = "●" * min(int(pct / 10) + 1, 10) + "○" * max(0, 10 - int(pct / 10) - 1)
    return (
        _BT_TERMINAL_CSS
        + f"""<div class="bt-terminal">
<span style="color:#ff6b35;font-size:13px"><b>{spinner} Backtest Prophet en cours…</b></span><br>
<span style="color:#00d4ff">{dots}</span>
<span style="color:#3fb950;margin-left:10px"><b>{pct:.1f}%</b></span>
<span style="color:#8b949e"> — {count} / {total} prédictions</span>
<br><br>
{log_html}
</div>"""
    )


with tab_backtest:
    st.subheader("Qualité des modèles — Rolling Backtest")

    # ── Sélecteur ticker + bouton reload ──────────────────────────────────────
    col_sel, col_rld = st.columns([5, 1])
    with col_sel:
        bt_ticker = st.selectbox(
            "Ticker",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD"],
            key="bt_ticker",
        )
    with col_rld:
        st.write("")
        if st.button("🔄 Recharger", key="bt_reload"):
            _load_backtest_detail.clear()
            _load_backtest_summary.clear()
            st.rerun()

    # ── Panneau de configuration + bouton Lancer ──────────────────────────────
    detail_df = _load_backtest_detail(bt_ticker)
    summary   = _load_backtest_summary(bt_ticker)
    has_data  = not detail_df.empty and bool(summary)

    with st.expander("⚙️ Lancer / reconfigurer le backtest", expanded=not has_data):
        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        with cfg_c1:
            bt_step = st.radio(
                "Vitesse (step)",
                options=[1, 5, 21],
                format_func=lambda x: {1: "Précis — ×1j", 5: "Rapide — ×5j", 21: "Express — ×21j"}[x],
                key="bt_step_radio",
            )
        with cfg_c2:
            bt_window = st.slider(
                "Fenêtre entraînement (j)", min_value=63, max_value=504,
                value=252, step=63, key="bt_window_slider",
            )
        with cfg_c3:
            bt_start = st.text_input("Date de début", value="2024-01-01", key="bt_start_date")

        launch_btn = st.button(
            f"🚀 Lancer le backtest — {bt_ticker}",
            type="primary", key="bt_launch",
            use_container_width=True,
        )

    # ── Exécution du backtest avec animation ──────────────────────────────────
    if launch_btn:
        try:
            from ml.evaluation.rolling_backtest import rolling_backtest_daily, save_backtest_results
            from ml.training.config import TrainingConfig
        except ImportError as e:
            st.error(
                f"**Impossible d'importer `ml.*` :** {e}\n\n"
                "En Docker, vérifiez que `./ml` est monté dans le container webapp."
            )
            st.stop()

        st.markdown(f"### Backtest `{bt_ticker}` en cours…")
        progress_bar = st.progress(0.0)
        terminal_box = st.empty()

        log_lines: list = []
        spin_state = [0]  # compteur mutable pour la closure

        def _on_progress(count: int, total: int, ticker: str, target_date) -> None:
            pct = count / total * 100
            spin = _BT_SPINNERS[spin_state[0] % len(_BT_SPINNERS)]
            spin_state[0] += 1
            log_lines.append(f"  {str(target_date):<12}  [{count:>4} / {total}]")
            progress_bar.progress(min(count / total, 1.0))
            terminal_box.markdown(
                _bt_terminal_html(spin, pct, count, total, log_lines),
                unsafe_allow_html=True,
            )

        try:
            cfg = TrainingConfig()
            detail_df, summary = rolling_backtest_daily(
                ticker=bt_ticker,
                train_window=bt_window,
                step=bt_step,
                start_date=bt_start,
                cfg=cfg,
                on_progress=_on_progress,
            )
            save_backtest_results(bt_ticker, detail_df, summary)
        except Exception as exc:
            terminal_box.error(f"Erreur pendant le backtest : {exc}")
            st.stop()

        # ── Fin : feedback + rechargement automatique ─────────────────────────
        progress_bar.progress(1.0)
        terminal_box.markdown(
            _BT_TERMINAL_CSS
            + f"""<div class="bt-terminal" style="border-color:#3fb95099;animation:none">
<span style="color:#3fb950;font-size:14px"><b>✅ Backtest terminé !</b></span><br><br>
<span style="color:#8b949e">  Ticker        : </span><span style="color:#00d4ff">{bt_ticker}</span><br>
<span style="color:#8b949e">  Prédictions   : </span><span style="color:#f0f6fc">{summary['n_predictions']}</span><br>
<span style="color:#8b949e">  MAE           : </span><span style="color:#f0f6fc">{summary['mae']}</span><br>
<span style="color:#8b949e">  MAPE          : </span><span style="color:#f0f6fc">{summary['mape_pct']}%</span><br>
<span style="color:#8b949e">  Dir. Accuracy : </span><span style="color:#f0f6fc">{summary['directional_accuracy_pct']}%</span><br><br>
<span style="color:#ff6b35">⟳ Rechargement des résultats…</span>
</div>""",
            unsafe_allow_html=True,
        )
        _load_backtest_detail.clear()
        _load_backtest_summary.clear()
        time.sleep(1.5)
        st.rerun()

    # ── Affichage des résultats existants ─────────────────────────────────────
    elif not has_data:
        st.info(
            f"Aucun artifact trouvé pour **{bt_ticker}**. "
            "Configurez et lancez le backtest via le panneau ci-dessus."
        )
    else:
        # ── Métriques principales ─────────────────────────────────────────────
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        with bm1:
            st.metric("MAE ($)", f"{summary.get('mae', '—')}")
        with bm2:
            st.metric("RMSE ($)", f"{summary.get('rmse', '—')}")
        with bm3:
            st.metric("MAPE (%)", f"{summary.get('mape_pct', '—'):.2f}%"
                      if isinstance(summary.get('mape_pct'), (int, float)) else "—")
        with bm4:
            da = summary.get("directional_accuracy_pct")
            st.metric(
                "Direction Accuracy",
                f"{da:.1f}%" if isinstance(da, (int, float)) else "—",
                delta=f"{da - 50:.1f}pts vs aléatoire" if isinstance(da, (int, float)) else None,
                delta_color="normal",
            )
        with bm5:
            cov = summary.get("interval_coverage_pct")
            st.metric(
                "Interval Coverage",
                f"{cov:.1f}%" if isinstance(cov, (int, float)) else "—",
            )

        # Méta-infos compactes
        st.caption(
            f"**{summary.get('n_predictions', '?')}** prédictions  ·  "
            f"fenêtre = {summary.get('train_window', '?')}j  ·  "
            f"step = {summary.get('step', '?')}j  ·  "
            f"depuis {summary.get('start_date', '?')}"
        )

        st.divider()

        # ── Graphique 1 : Actual vs Predicted ────────────────────────────────
        st.plotly_chart(
            _make_backtest_chart(detail_df, bt_ticker),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )

        # ── Graphique 2 : Erreur absolue + couverture interval ───────────────
        st.plotly_chart(
            _make_error_chart(detail_df, bt_ticker),
            use_container_width=True,
            config={"displayModeBar": True},
        )

        # ── Tableau récapitulatif (dernières 20 prédictions) ─────────────────
        st.divider()
        with st.expander("📋 Dernières prédictions (20 lignes)", expanded=False):
            display_cols = {
                "target_date":       "Date cible",
                "actual":            "Réel ($)",
                "yhat":              "Prédit ($)",
                "abs_error":         "Erreur abs ($)",
                "pct_error":         "Erreur (%)",
                "direction_correct": "Direction OK",
                "interval_hit":      "Dans l'intervalle",
            }
            df_show = detail_df.tail(20)[list(display_cols.keys())].copy()
            df_show.rename(columns=display_cols, inplace=True)
            df_show["Date cible"]         = df_show["Date cible"].astype(str)
            df_show["Réel ($)"]           = df_show["Réel ($)"].apply(lambda x: f"{x:,.2f}")
            df_show["Prédit ($)"]         = df_show["Prédit ($)"].apply(lambda x: f"{x:,.2f}")
            df_show["Erreur abs ($)"]     = df_show["Erreur abs ($)"].apply(lambda x: f"{x:,.2f}")
            df_show["Erreur (%)"]         = df_show["Erreur (%)"].apply(lambda x: f"{x:.2f}%")
            df_show["Direction OK"]       = df_show["Direction OK"].map({1: "✅", 0: "❌"})
            df_show["Dans l'intervalle"]  = df_show["Dans l'intervalle"].map({1: "✅", 0: "❌"})
            st.dataframe(df_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB CONSEIL — Analyse directionnelle basée sur Prophet (J+1 et J+30)
# ══════════════════════════════════════════════════════════════════════════════

def _make_direction_signal_bars(
    df_pred_daily: pd.DataFrame | None,
    df_pred_monthly: pd.DataFrame | None,
    ticker: str,
) -> go.Figure:
    """
    Graphique de direction Prophet — SANS valeur de prix.
    Barres centrées sur 0 : (prob_up - 50%) * 2
      > 0 = signal HAUSSE  (cyan)
      < 0 = signal BAISSE  (orange)
    Les deux séries (daily et monthly) sont superposées.
    """
    fig = go.Figure()

    # Zones fond
    fig.add_hrect(y0=0,   y1=100,  fillcolor="rgba(0,212,255,0.04)",   line_width=0)
    fig.add_hrect(y0=-100, y1=0,   fillcolor="rgba(255,107,53,0.04)",  line_width=0)

    # ── Série daily ──────────────────────────────────────────────────────────
    if df_pred_daily is not None and "prob_up" in df_pred_daily.columns:
        probs_d  = df_pred_daily["prob_up"].fillna(0.5)
        signal_d = (probs_d - 0.5) * 200   # centré, mis à l'échelle [-100, +100]
        colors_d = ["#00d4ff" if s >= 0 else "#ff6b35" for s in signal_d]
        fig.add_trace(go.Bar(
            x=df_pred_daily["date"],
            y=signal_d,
            name="Signal daily (J+1)",
            marker_color=colors_d,
            opacity=0.85,
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Signal : <b>%{y:+.1f}</b>  (prob %{customdata:.1f}%)"
                "<extra>Daily</extra>"
            ),
            customdata=probs_d * 100,
        ))

    # ── Série monthly ─────────────────────────────────────────────────────────
    if df_pred_monthly is not None and "prob_up" in df_pred_monthly.columns:
        probs_m  = df_pred_monthly["prob_up"].fillna(0.5)
        signal_m = (probs_m - 0.5) * 200
        colors_m = ["rgba(0,212,255,0.5)" if s >= 0 else "rgba(255,107,53,0.5)" for s in signal_m]
        fig.add_trace(go.Scatter(
            x=df_pred_monthly["date"],
            y=signal_m,
            name="Signal mensuel (J+30)",
            line=dict(color="#b45aff", width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=7, color="#b45aff", symbol="diamond"),
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Signal : <b>%{y:+.1f}</b>  (prob %{customdata:.1f}%)"
                "<extra>Mensuel</extra>"
            ),
            customdata=probs_m * 100,
        ))

    # Seuils
    fig.add_hline(y=0,   line=dict(color="rgba(255,255,255,0.25)", width=1.5))
    fig.add_hline(y=10,  line=dict(color="rgba(0,212,255,0.4)", dash="dot", width=1),
                  annotation_text="Seuil HAUSSE", annotation_font_color="rgba(0,212,255,0.7)",
                  annotation_position="top left")
    fig.add_hline(y=-10, line=dict(color="rgba(255,107,53,0.4)", dash="dot", width=1),
                  annotation_text="Seuil BAISSE", annotation_font_color="rgba(255,107,53,0.7)",
                  annotation_position="bottom left")

    # Ligne "aujourd'hui"
    today_str = pd.Timestamp.today().normalize().isoformat()
    fig.add_shape(
        type="line", x0=today_str, x1=today_str, y0=0, y1=1, yref="paper",
        line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1.5),
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — Force directionnelle Prophet  "
                 "<span style='color:#00d4ff'>▲ Hausse</span>  /  "
                 "<span style='color:#ff6b35'>▼ Baisse</span>",
            font=dict(size=17), x=0.01,
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Signal directionnel (centré sur 0)",
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        barmode="overlay",
        height=380,
        plot_bgcolor="rgba(8,12,24,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=80, b=50),
    )
    return fig


def _make_prob_trend_chart(
    df_pred_daily: pd.DataFrame | None,
    ticker: str,
) -> go.Figure:
    """
    Graphique d'évolution de la prob_up Prophet sur le forecast daily
    pour visualiser la tendance directionnelle dans le temps.
    """
    fig = go.Figure()

    if df_pred_daily is not None and "prob_up" in df_pred_daily.columns:
        probs = df_pred_daily["prob_up"].fillna(0.5) * 100

        # Zone de couleur de fond selon le signal
        fig.add_hrect(y0=55, y1=100, fillcolor="rgba(0,212,255,0.05)", line_width=0)
        fig.add_hrect(y0=0,  y1=45,  fillcolor="rgba(255,107,53,0.05)", line_width=0)

        # Courbe de prob_up
        colors_line = [_gauge_color(p / 100) for p in probs]
        fig.add_trace(go.Scatter(
            x=df_pred_daily["date"],
            y=probs,
            mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=6, color=colors_line),
            name="Prob. hausse Prophet",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Prob. hausse : <b>%{y:.1f}%</b><extra></extra>",
        ))

    # Seuils
    fig.add_hline(y=55, line=dict(color="rgba(0,212,255,0.5)", dash="dot", width=1.5),
                  annotation_text="55% — HAUSSE", annotation_font_color="rgba(0,212,255,0.7)",
                  annotation_position="top left")
    fig.add_hline(y=50, line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1),)
    fig.add_hline(y=45, line=dict(color="rgba(255,107,53,0.5)", dash="dot", width=1.5),
                  annotation_text="45% — BAISSE", annotation_font_color="rgba(255,107,53,0.7)",
                  annotation_position="bottom left")

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — Évolution de la probabilité de hausse (Prophet daily)",
            font=dict(size=16), x=0.01,
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Prob. hausse (%)", range=[0, 100],
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        height=300,
        plot_bgcolor="rgba(8,12,24,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=55, r=20, t=60, b=50),
        showlegend=False,
    )
    return fig


with tab_conseil:
    st.subheader("💡 Conseil — Analyse directionnelle Prophet")
    st.caption(
        "Analyse basée sur le modèle **Prophet** : tendance court terme (J+1) "
        "et moyen terme (J+30). Les deux horizons sont récupérés puis comparés."
    )

    # ── Sélection ticker + bouton ──────────────────────────────────────────────
    cs_col1, cs_col2 = st.columns([3, 1])
    with cs_col1:
        cs_ticker = st.selectbox(
            "Ticker",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD"],
            key="conseil_ticker_select",
        )
    with cs_col2:
        st.write("")
        st.write("")
        cs_btn = st.button("📡 Analyser", use_container_width=True, type="primary", key="conseil_btn")

    if cs_btn:
        with st.spinner(f"Récupération des prévisions Prophet pour {cs_ticker}…"):
            cs_data = {}
            for cs_mode in ("daily", "monthly"):
                try:
                    r = requests.post(
                        f"{SERVING_URL}/predict",
                        json={"ticker": cs_ticker, "mode": cs_mode},
                        timeout=30,
                    )
                    r.raise_for_status()
                    cs_data[cs_mode] = r.json()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de joindre le service serving (http://serving:8080).")
                    st.stop()
                except Exception as e:
                    st.warning(f"⚠️ Impossible de charger le modèle '{cs_mode}' pour {cs_ticker} : {e}")

            st.session_state["last_conseil"] = {
                "ticker": cs_ticker,
                "data":   cs_data,
            }

    if "last_conseil" in st.session_state:
        cs = st.session_state["last_conseil"]
        tk_cs    = cs["ticker"]
        cs_daily = cs["data"].get("daily")
        cs_month = cs["data"].get("monthly")

        # Extraire les DataFrames
        df_daily = None
        df_month = None
        if cs_daily:
            df_daily = pd.DataFrame(cs_daily["predictions"])
            df_daily["date"] = pd.to_datetime(df_daily["date"])
        if cs_month:
            df_month = pd.DataFrame(cs_month["predictions"])
            df_month["date"] = pd.to_datetime(df_month["date"])

        dir_daily = cs_daily["direction"] if cs_daily else None
        dir_month = cs_month["direction"] if cs_month else None

        # Probabilités
        prob_daily = None
        prob_month = None
        if df_daily is not None and "prob_up" in df_daily.columns:
            vals = df_daily["prob_up"].dropna()
            if len(vals):
                prob_daily = float(vals.iloc[-1])
        if df_month is not None and "prob_up" in df_month.columns:
            vals = df_month["prob_up"].dropna()
            if len(vals):
                prob_month = float(vals.iloc[-1])

        # ── Helpers locaux ────────────────────────────────────────────────────
        def _dir_label(d) -> str:
            return "—" if d is None else ("▲ HAUSSE" if d else "▼ BAISSE")

        def _consensus_label(d1, d2) -> str:
            if d1 is None and d2 is None:
                return "—"
            if d1 == d2:
                return ("▲ HAUSSIER" if d1 else "▼ BAISSIER") + " ✅"
            return "⚡ MIXTE"

        # ── Métriques direction ───────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Ticker", tk_cs)
        with mc2:
            st.metric("Tendance J+1", _dir_label(dir_daily))
        with mc3:
            st.metric("Tendance J+30", _dir_label(dir_month))
        with mc4:
            st.metric("Consensus", _consensus_label(dir_daily, dir_month))

        st.divider()

        # ── Jauges Prophet (direction, pas de prix) ───────────────────────────
        if prob_daily is not None or prob_month is not None:
            gd1, gd2 = st.columns(2)
            with gd1:
                if prob_daily is not None:
                    st.plotly_chart(
                        _make_gauge("Prob. hausse J+1 (Prophet)", prob_daily),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
            with gd2:
                if prob_month is not None:
                    st.plotly_chart(
                        _make_gauge("Prob. hausse J+30 (Prophet)", prob_month),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
            st.caption(
                "🔵 Cyan = HAUSSE (prob > 55 %)  ·  "
                "🟠 Orange = BAISSE (prob < 45 %)  ·  "
                "⬜ Gris = NEUTRE"
            )
            st.divider()

        # ── Graphique force directionnelle (barres centrées sur 0) ───────────
        st.plotly_chart(
            _make_direction_signal_bars(df_daily, df_month, tk_cs),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )

        # ── Évolution de la prob_up daily dans le temps ───────────────────────
        if df_daily is not None and "prob_up" in df_daily.columns:
            st.plotly_chart(
                _make_prob_trend_chart(df_daily, tk_cs),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Tableau de direction (sans prix) ──────────────────────────────────
        st.divider()
        rows_cmp = []
        if df_daily is not None and not df_daily.empty:
            row = {"Horizon": "J+1 (daily)", "Direction": _dir_label(dir_daily)}
            if prob_daily is not None:
                row["Prob. hausse"] = f"{prob_daily*100:.1f}%"
            if "prob_up" in df_daily.columns:
                avg_d = float(df_daily["prob_up"].mean()) * 100
                row["Prob. moy. sur période"] = f"{avg_d:.1f}%"
            rows_cmp.append(row)
        if df_month is not None and not df_month.empty:
            row = {"Horizon": "J+30 (mensuel)", "Direction": _dir_label(dir_month)}
            if prob_month is not None:
                row["Prob. hausse"] = f"{prob_month*100:.1f}%"
            if "prob_up" in df_month.columns:
                avg_m = float(df_month["prob_up"].mean()) * 100
                row["Prob. moy. sur période"] = f"{avg_m:.1f}%"
            rows_cmp.append(row)
        if rows_cmp:
            st.markdown("#### Récapitulatif directionnel Prophet")
            st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

        # ── Notification Agent IA ─────────────────────────────────────────────
        st.divider()
        st.markdown("#### 🤖 Envoyer une analyse par email")
        notif_email = st.text_input(
            "Adresse email du destinataire",
            placeholder="investisseur@exemple.com",
            key="conseil_notif_email",
        )
        if st.button("📧 Envoyer l'analyse via l'Agent IA", key="conseil_agent", type="primary"):
            if not notif_email:
                st.warning("Veuillez saisir une adresse email.")
            else:
                # Inclure les signaux LSTM s'ils sont disponibles pour ce ticker
                lstm_payload = None
                last_lstm = st.session_state.get("last_lstm_prediction", {})
                if last_lstm.get("ticker") == tk_cs:
                    lstm_payload = {
                        "prob_1d":   last_lstm.get("prob_1d",  0.5),
                        "prob_7d":   last_lstm.get("prob_7d",  0.5),
                        "prob_30d":  last_lstm.get("prob_30d", 0.5),
                        "signal_1d":  last_lstm.get("signal_1d",  "NEUTRE"),
                        "signal_7d":  last_lstm.get("signal_7d",  "NEUTRE"),
                        "signal_30d": last_lstm.get("signal_30d", "NEUTRE"),
                    }

                payload = {
                    "user_email":        notif_email,
                    "ticker":            tk_cs,
                    "direction_daily":   dir_daily,
                    "direction_monthly": dir_month,
                    "prob_daily":        prob_daily,
                    "prob_month":        prob_month,
                    "lstm":              lstm_payload,
                }
                with st.spinner("L'agent génère et envoie l'email…"):
                    try:
                        r = requests.post(f"{AGENT_URL}/notify", json=payload, timeout=60)
                        r.raise_for_status()
                        st.success(f"Email envoyé à **{notif_email}** — le destinataire peut maintenant valider ou contester l'analyse.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Impossible de joindre l'agent (http://agent:8083). Vérifiez que le service est démarré.")
                    except Exception as e:
                        st.error(f"Erreur agent : {e}")

    else:
        st.info("👆 Sélectionnez un ticker et cliquez sur **Analyser** pour obtenir le conseil directionnel.")
