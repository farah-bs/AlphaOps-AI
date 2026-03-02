import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatMistralAI(
    model="codestral-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.1,
    max_tokens=2048
)

sql_prompt = ChatPromptTemplate.from_template("""
Tu es un expert SQL PostgreSQL. Convertis la question en SQL, en respectant les règles suivantes :
                                            
                                              - Schéma : public
                                              - Utilise uniquement les tables : dim_tickers, dim_time, fact_ohlcv

                                              Question : {question}

                                              Réponds en SQL puis tu peux ajouter une brève explication de ta requête, et aussi des données.
""")

sql_chain = sql_prompt | llm