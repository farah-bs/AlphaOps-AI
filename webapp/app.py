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
N8N_WEBHOOK  = "http://n8n:5678/webhook/alphaops-alert"

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


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_predict, tab_backtest = st.tabs(["💬 Chatbot", "🔮 Prédiction", "📊 Backtest"])

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


# ── Onglet Prédiction ─────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Prédiction de marché")
    st.caption("Modèle Prophet servi via le container serving.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker = st.selectbox(
            "Ticker",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "BTC-USD"],
        )
    with col2:
        mode = st.radio("Horizon", ["daily", "monthly"], horizontal=True)
    with col3:
        st.write("")
        st.write("")
        predict_btn = st.button("🔮 Prédire", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("Interrogation du serving..."):
            try:
                resp = requests.post(
                    f"{SERVING_URL}/predict",
                    json={"ticker": ticker, "mode": mode},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state["last_prediction"] = {
                    "ticker":      ticker,
                    "mode":        mode,
                    "direction":   data["direction"],
                    "predictions": data["predictions"],
                }
            except requests.exceptions.ConnectionError:
                st.error("Impossible de joindre le service serving (http://serving:8080).")
            except Exception as e:
                st.error(f"Erreur : {e}")

    if "last_prediction" in st.session_state:
        last      = st.session_state["last_prediction"]
        direction = last["direction"]
        tk        = last["ticker"]

        df_pred = pd.DataFrame(last["predictions"])
        df_pred["date"] = pd.to_datetime(df_pred["date"])

        # ── Données historiques ──────────────────────────────────────────────
        df_hist = pd.DataFrame()
        delta_pct = None
        try:
            df_hist = fetch_history(tk, days=90)
            if not df_hist.empty:
                last_close = float(df_hist["adj_close"].iloc[-1])
                first_pred = float(df_pred["yhat"].iloc[0])
                delta_pct  = (first_pred - last_close) / last_close * 100
        except Exception:
            pass

        # ── Métriques ────────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric(
                "Direction",
                "▲ HAUSSE" if direction else "▼ BAISSE",
                delta=f"{delta_pct:+.2f}%" if delta_pct is not None else None,
                delta_color="normal",
            )
        with mc2:
            st.metric("Ticker", tk)
        with mc3:
            st.metric("Horizon", "J+1" if last["mode"] == "daily" else "J+30")
        with mc4:
            if not df_hist.empty:
                st.metric(
                    "Dernier cours",
                    f"${df_hist['adj_close'].iloc[-1]:,.2f}",
                )

        st.divider()

        # ── Chart 2D ─────────────────────────────────────────────────────────
        if not df_hist.empty:
            st.plotly_chart(
                make_2d_chart(df_hist, df_pred, tk, direction),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 2}},
            )
        else:
            # Fallback sans historique
            fig_simple = go.Figure()
            fig_simple.add_trace(go.Scatter(
                x=df_pred["date"], y=df_pred["yhat"],
                mode="lines+markers",
                line=dict(color="#ff6b35", width=2),
                name="Prévision",
            ))
            fig_simple.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_simple, use_container_width=True)

        # ── Chart 3D ─────────────────────────────────────────────────────────
        if not df_hist.empty and len(df_pred) > 1:
            with st.expander("🧊 Vue 3D — Tunnel de prix", expanded=True):
                st.caption(
                    "Surface bleue = historique (bas=low, médian=clôture, haut=high) · "
                    "Surface orange = forecast (bas=yhat_lower, médian=yhat, haut=yhat_upper) · "
                    "Plan blanc = aujourd'hui"
                )
                st.plotly_chart(
                    make_3d_chart(df_hist, df_pred, tk),
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True},
                )

        # ── Tableau détail ───────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Détail des prédictions")

        df_display = df_pred[["date", "yhat", "yhat_lower", "yhat_upper"]].copy()
        df_display.columns = ["Date", "Prévision ($)", "Borne basse ($)", "Borne haute ($)"]
        df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%d %b %Y")
        for col in ["Prévision ($)", "Borne basse ($)", "Borne haute ($)"]:
            df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # ── Notification n8n ─────────────────────────────────────────────────
        st.divider()
        if st.button("🔔 Notifier via n8n"):
            try:
                r = requests.post(N8N_WEBHOOK, json=last, timeout=10)
                r.raise_for_status()
                st.success("Notification envoyée à n8n.")
            except Exception as e:
                st.error(f"Erreur notification : {e}")


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
