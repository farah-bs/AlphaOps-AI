import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agents.agent import ask_agent

SERVING_URL = "http://serving:8080"
N8N_WEBHOOK  = "http://n8n:5678/webhook/alphaops-alert"

st.set_page_config(page_title="AlphaOps AI", page_icon="📈", layout="wide")
st.title("📈 AlphaOps AI")

tab_chat, tab_predict = st.tabs(["Chatbot", "Prédiction"])

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
    st.subheader("Prédiction de direction")
    st.caption("Interroge le modèle Prophet via le service de serving.")

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
        predict_btn = st.button("Prédire", use_container_width=True)

    if predict_btn:
        with st.spinner("Appel au service de serving..."):
            try:
                resp = requests.post(
                    f"{SERVING_URL}/predict",
                    json={"ticker": ticker, "mode": mode},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                direction = data["direction"]
                preds     = data["predictions"]

                if direction:
                    st.success(f"Direction prévue : HAUSSE pour {ticker}")
                else:
                    st.error(f"Direction prévue : BAISSE pour {ticker}")

                df_pred = pd.DataFrame(preds)
                df_pred.columns = ["Date", "Prévision", "Borne basse", "Borne haute"]
                st.dataframe(df_pred, use_container_width=True)

                st.session_state["last_prediction"] = {
                    "ticker":    ticker,
                    "mode":      mode,
                    "direction": direction,
                    "predictions": preds,
                }

            except requests.exceptions.ConnectionError:
                st.error("Impossible de joindre le service serving (http://serving:8080). Est-il démarré ?")
            except Exception as e:
                st.error(f"Erreur : {e}")

    if "last_prediction" in st.session_state:
        st.divider()
        last = st.session_state["last_prediction"]
        st.markdown(f"**Dernière prédiction** : `{last['ticker']}` — `{last['mode']}` — {'HAUSSE' if last['direction'] else 'BAISSE'}")

        if st.button("Notifier via n8n"):
            try:
                r = requests.post(N8N_WEBHOOK, json=last, timeout=10)
                r.raise_for_status()
                st.success("Notification envoyée à n8n.")
            except Exception as e:
                st.error(f"Erreur notification : {e}")