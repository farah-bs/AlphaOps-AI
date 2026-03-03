import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agents.agent import ask_agent

st.set_page_config(page_title="AlphaOps AI - NL2SQL Chat", page_icon="📈", layout="wide")

st.title("📈 AlphaOps AI — Chatbot Stocks")

# Session state: historique messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input utilisateur
prompt = st.chat_input("Pose une question (ex: 'Prix NVDA hier ?')")

if prompt:
    # Affiche message user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Appel agent
    with st.chat_message("assistant"):
        with st.spinner("Je génère la requête SQL et je consulte la base..."):
            out = ask_agent(prompt)

        # 1) SQL généré
        sql = out.get("sql_query", "")
        if sql:
            st.subheader("🧾 SQL généré")
            st.code(sql, language="sql")

        # 2) Validation
        val = out.get("validation", {})
        if val:
            st.subheader("✅ Validation")
            st.json(val)

        # 3) Résultat brut
        st.subheader("📦 Résultat brut")
        st.write(out.get("result"))

        # 4) Réponse “humaine” si dispo
        if "answer" in out and out["answer"]:
            st.subheader("🗣️ Réponse")
            st.markdown(out["answer"])

        assistant_msg = "✅ Requête traitée. (Regarde le SQL et la réponse ci-dessus.)"

    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})