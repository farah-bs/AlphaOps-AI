import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agents.agent import ask_agent

st.set_page_config(page_title="AlphaOps AI - Chat", page_icon="📈", layout="wide")

st.title("📈 AlphaOps AI — Chatbot Stocks")
st.caption("Exemples : 'Dernier prix Apple', 'OHLC NVIDIA hier', 'Volume TSLA la semaine dernière'")

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