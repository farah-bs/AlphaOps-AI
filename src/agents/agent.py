import re
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from src.validators.sql_validator import SQLValidator
import os
from dotenv import load_dotenv
import re
from decimal import Decimal
from datetime import date
from src.db.connection import get_engine

load_dotenv()

engine = get_engine()
db = SQLDatabase(engine)
llm = ChatMistralAI(model="codestral-latest", api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.1)

class AgentState(TypedDict):
    input: str
    sql_query: str
    validation: dict
    result: str
    messages: Annotated[list, operator.add]
    db_schema: str
    answer: str
    symbol: str

def get_schema(state):
    schema = db.get_table_info(["dim_tickers", "dimtime", "fact_ohlcv"])
    return {"db_schema": schema}

def generate_sql(state):
    symbol = state.get("symbol", "")
    
    if symbol == "__GENERAL__" or not symbol:
        prompt = ChatPromptTemplate.from_template("""
Tu es un assistant Text-to-SQL pour PostgreSQL.

Tables autorisées :
- dim_tickers(symbol, name, market, sector, first_date, last_date, avg_volume, created_at)
- dimtime(date, year, month, day, quarter, day_of_week, is_weekend, is_month_end)
- fact_ohlcv(symbol, date, open_price, high_price, low_price, close_price, volume, adj_close, volatility)

Cette question est GÉNÉRALE (pas de ticker spécifique demandé).

Règles pour questions générales:
- Retourne les données pour PLUSIEURS tickers (les plus pertinents)
- Utilise une sous-requête pour récupérer la dernière date disponible
- LIMIT 10 pour montrer un aperçu
- Trie par date DESC puis par symbol
- Ne PAS filtrer sur un symbol spécifique

Exemples de requêtes générales:
- "Derniers prix" → derniers close_price de chaque ticker à leur date la plus récente
- "Top volumes" → tickers avec les plus gros volumes récemment
- "Aperçu du marché" → résumé des principaux tickers

Schéma: {db_schema}

Question: {input}

Réponds uniquement avec le SQL.
""")
    else:
        prompt = ChatPromptTemplate.from_template("""
Tu es un assistant Text-to-SQL pour PostgreSQL.

Tables autorisées :
- dim_tickers(symbol, name, market, sector, first_date, last_date, avg_volume, created_at)
- dimtime(date, year, month, day, quarter, day_of_week, is_weekend, is_month_end)
- fact_ohlcv(symbol, date, open_price, high_price, low_price, close_price, volume, adj_close, volatility)

Règles:
- Génère UNE SEULE requête SELECT PostgreSQL safe.
- Filtre sur le ticker: {symbol}
- SELECT uniquement (WITH...SELECT autorisé).
- LIMIT <= 50.
- Pour "hier"/"dernier"/"latest": WHERE date < CURRENT_DATE ORDER BY date DESC LIMIT 1
- Si nom de société donné, mapper via dim_tickers.name
- Ne jamais SELECT *

Mapping colonnes:
- "prix" → close_price
- "volume" → volume  
- "volatilité" → volatility
- "ohlc"/"détails" → open_price, high_price, low_price, close_price

Schéma: {db_schema}

Question: {input}
Ticker: {symbol}

Réponds uniquement avec le SQL.
""")

    chain = prompt | llm
    
    if symbol == "__GENERAL__" or not symbol:
        params = {
            "db_schema": state["db_schema"],
            "input": state["input"]
        }
    else:
        params = {
            "db_schema": state["db_schema"],
            "input": state["input"],
            "symbol": symbol
        }

    sql = chain.invoke(params).content.strip()

    # enlever ```sql ... ```
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    return {"sql_query": sql}

def validate_sql(state):
    validator = SQLValidator()
    v = validator.validate(state["sql_query"])
    if not v.get("is_valid"):
        return {"validation": v, "result": f"Requête refusée: {v.get('reason')}"}
    return {"validation": v}


def execute_sql(state):
    if not state.get('validation', {}).get('is_valid'):
        return {"result": state['validation']['reason']}
    try:
        result = db.run(state['sql_query'])
        return {"result": result}
    except Exception as e:
        return {"result": f"Sheesh execution error: {str(e)}"}
    
def format_answer(state):
    raw = state.get("result", "")
    question = state.get("input", "")
    sql = state.get("sql_query", "")

    if raw.startswith("Requête refusée") or raw.startswith("Sheesh") or "error" in raw.lower():
        return {"answer": raw}

    if not raw or raw == "[]":
        return {"answer": "Aucune donnée trouvée pour cette question."}

    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant financier expert et amical. Réponds en français de manière naturelle, comme si tu parlais à un ami.

Question de l'utilisateur : {question}
Données récupérées : {raw}

Consignes IMPORTANTES :
- Réponds de façon conversationnelle et naturelle, comme ChatGPT ou Claude
- NE MONTRE JAMAIS le format brut des données (pas de tuples, pas de Decimal(), pas de datetime.date())
- Utilise le **gras** pour mettre en valeur les chiffres importants
- Pour les prix : formate avec 2 décimales et ajoute "$" (ex: **263.99 $**)
- Pour les dates : écris en français naturel (ex: "le 2 mars 2026" ou "hier")
- Pour les volumes : utilise des séparateurs de milliers (ex: **1 234 567**)
- Si plusieurs résultats, présente-les clairement avec une liste ou un tableau
- Ajoute un petit contexte si pertinent (ex: "Le cours de clôture de...")
- Sois concis mais informatif
- NE MENTIONNE JAMAIS le SQL ou la technique derrière
""")

    chain = prompt | llm
    answer = chain.invoke({
        "question": question,
        "raw": raw
    }).content.strip()

    return {"answer": answer}
    
FRENCH_STOPWORDS = {
    "LE", "LA", "LES", "UN", "UNE", "DES", "DE", "DU", "AU", "AUX",
    "MON", "TON", "SON", "MA", "TA", "SA", "MES", "TES", "SES",
    "CE", "CET", "CETTE", "CES", "QUEL", "QUELLE", "QUELS",
    "JE", "TU", "IL", "ELLE", "ON", "NOUS", "VOUS", "ILS", "ELLES",
    "MOI", "TOI", "LUI", "EUX", "LEUR", "LEURS",
    "QUI", "QUE", "QUOI", "DONT", "OU", "AVEC", "SANS", "POUR", "PAR",
    "DANS", "SUR", "SOUS", "ENTRE", "VERS", "CHEZ",
    "ET", "OU", "MAIS", "DONC", "NI", "CAR", "SI", "PUIS",
    "TOUT", "TOUS", "TOUTE", "TOUTES", "AUTRE", "AUTRES", "MEME",
    "ETRE", "AVOIR", "FAIRE", "DIRE", "ALLER", "VOIR", "VENIR", "POUVOIR",
    "EST", "SONT", "SUIS", "ES", "SOMMES", "ETES", "ETAIT", "SERA",
    "AI", "AS", "AVONS", "AVEZ", "ONT", "AURA", "AVAIT",
    "FAIT", "FAIS", "FONT", "FERA", "FAISAIT",
    "DONNE", "DONNER", "MONTRE", "MONTRER", "AIDE", "AIDER",
    "PRIX", "COURS", "ACTION", "STOCK", "BOURSE", "VALEUR",
    "HIER", "AVANT", "APRES", "JOUR", "MOIS", "ANNEE", "DATE",
    "PLUS", "MOINS", "TRES", "BIEN", "MAL", "PAS", "JAMAIS", "ENCORE",
    "HAUT", "BAS", "GRAND", "PETIT", "VIEUX", "NEUF", "NOUVEAU",
    "PREMIER", "DERNIER", "SEUL", "DEUX", "TROIS", "DIX", "CENT",
    "QUEL", "QUELLE", "COMMENT", "QUAND", "COMBIEN", "POURQUOI",
    "OUI", "NON", "PEUT", "PEUX", "VEUT", "VEUX", "FAUT", "DOIT",
    "AUSSI", "COMME", "AINSI", "ALORS", "DEPUIS", "PENDANT",
    "BOURSIER", "BOURSIERS", "VOLUME", "OHLC", "OPEN", "HIGH", "LOW", "CLOSE"
}

def resolve_symbol(state):
    question = state["input"].upper()
    
    candidates = re.findall(r"\b[A-Z]{1,5}(?:-[A-Z]{2,5})?\b", question)
    
    valid_candidates = [c for c in candidates if c not in FRENCH_STOPWORDS]
    
    for candidate in valid_candidates:
        check_sql = f"SELECT symbol FROM dim_tickers WHERE symbol = '{candidate}' LIMIT 1"
        try:
            res = db.run(check_sql)
            if res and len(res) > 2: 
                return {"symbol": candidate}
        except:
            pass
    
    q = state["input"].strip()
    words = re.findall(r"\b[A-Za-z]{3,}\b", q)
    for word in words:
        if word.upper() in FRENCH_STOPWORDS:
            continue
        search_sql = f"SELECT symbol FROM dim_tickers WHERE LOWER(name) LIKE LOWER('%{word}%') LIMIT 1"
        try:
            res = db.run(search_sql)
            if res and "(" in res:  
                match = re.search(r"\('([A-Z]+)'\)", res)
                if match:
                    return {"symbol": match.group(1)}
        except:
            pass
    
    return {"symbol": "__GENERAL__"}

def ask_agent(question: str):
    result = app.invoke({"input": question})

    return {
        "sql": result.get("sql_query"),
        "validation": result.get("validation"),
        "raw_result": result.get("result"),
        "answer": result.get("answer") or "Aucune réponse."
    }



workflow = StateGraph(AgentState)

workflow.add_node("get_schema", get_schema)
workflow.add_node("resolve_symbol", resolve_symbol)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("validate_sql", validate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("format_answer", format_answer)

workflow.set_entry_point("get_schema")

workflow.add_edge("get_schema", "resolve_symbol")
workflow.add_edge("resolve_symbol", "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")

workflow.add_conditional_edges(
    "validate_sql",
    lambda s: "execute_sql" if s["validation"]["is_valid"] else "format_answer"
)

workflow.add_edge("execute_sql", "format_answer")
workflow.add_edge("format_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({"input": "Prix NVDA hier ?"})
    print("SQL généré:", result.get("sql_query", "No SQL"))
    print("Réponse:", result.get("answer", "No answer"))
    print("Résultat brut:", result.get("result", "No raw result"))