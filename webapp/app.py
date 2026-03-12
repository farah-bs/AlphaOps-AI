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

tab_chat, tab_predict = st.tabs(["💬 Chatbot", "🔮 Prédiction"])

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
