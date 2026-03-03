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
    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant Text-to-SQL pour PostgreSQL.

Tables autorisées uniquement :
- dim_tickers(symbol, name, market, sector, first_date, last_date, avg_volume, created_at)
- dimtime(date, year, month, day, quarter, day_of_week, is_weekend, is_month_end)
- fact_ohlcv(symbol, date, open_price, high_price, low_price, close_price, volume, adj_close, volatility)

Règles:
- Génère UNE SEULE requête SELECT PostgreSQL safe.
- Utilise {symbol} (obligatoire) dans les filtres.
- SELECT uniquement (WITH...SELECT autorisé).
- Toujours utiliser LIMIT <= 50.
- Pour "hier"/"dernier"/"latest": utiliser la dernière date disponible avant CURRENT_DATE:
  WHERE symbol = 'XXX' AND date < CURRENT_DATE ORDER BY date DESC LIMIT 1
- Si l'utilisateur donne un nom (ex: "NVIDIA"), mapper via dim_tickers.name pour retrouver symbol.
- Ne jamais utiliser SELECT *.
                                              
Si la question mentionne :
- "prix" → sélectionner close_price
- "volume" → sélectionner volume
- "volatilité" → sélectionner volatility
- "ohlc" ou "détails" → sélectionner open, high, low, close

Schéma technique:
{db_schema}

Question user: {input}
Ticker détecté: {symbol}

Réponds uniquement avec le SQL.
""")

    chain = prompt | llm

    sql = chain.invoke({
        "db_schema": state["db_schema"],
        "input": state["input"],
        "symbol": state["symbol"]
    }).content.strip()

    # enlever ```sql ... ```
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    return {"sql_query": sql}

def validate_sql(state):
    validator = SQLValidator()
    return {"validation": validator.validate(state['sql_query'])}


def execute_sql(state):
    if not state.get('validation', {}).get('is_valid'):
        return {"result": state['validation']['reason']}
    try:
        result = db.run(state['sql_query'])
        return {"result": result}
    except Exception as e:
        return {"result": f"Sheesh execution error: {str(e)}"}
    
def format_answer(state):
    raw = state.get("result")
    question = state.get("input", "")

    if isinstance(raw, str):
        return {"answer": raw}

    if not raw:
        return {"answer": "Aucune donnée trouvée."}

    row = raw[0]

    if len(row) == 2:
        d, value = row

        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")

        if isinstance(value, Decimal):
            value = float(value)

        return {
            "answer": f"📅 {d}\n💰 Valeur : {value:.2f}"
        }

    if len(row) == 7:
        (
            d,
            open_p,
            high_p,
            low_p,
            close_p,
            volume,
            volat
        ) = row

        return {
            "answer": f"""
📅 Date : {d.strftime("%Y-%m-%d")}

📊 OHLC :
- Open : {float(open_p):.2f} USD
- High : {float(high_p):.2f} USD
- Low  : {float(low_p):.2f} USD
- Close: {float(close_p):.2f} USD

📈 Volume : {volume:,}
📉 Volatilité : {float(volat)*100:.2f}%
""".strip()
        }

    if len(raw) > 1:
        return {
            "answer": f"Requête retournée {len(raw)} lignes. Exemple première ligne: {row}"
        }

    return {"answer": str(raw)}
    
def resolve_symbol(state):
    question = state["input"].upper()

    m = re.search(r"\b[A-Z]{1,5}(?:-[A-Z]{2,5})?\b", question)
    if m:
        sym = m.group(0)
        return {"symbol": sym}

    q = state["input"].strip()
    sql = """
    SELECT symbol
    FROM dim_tickers
    WHERE LOWER(name) LIKE LOWER(:q)
    LIMIT 1
    """
    res = db.run(sql, {"q": f"%{q}%"})
    if res:
        return {"symbol": res[0][0]}

    return {"symbol": ""}

workflow = StateGraph(AgentState)
workflow.add_node("get_schema", get_schema)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("validate_sql", validate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("format_answer", format_answer)
workflow.add_node("resolve_symbol", resolve_symbol)
workflow.set_entry_point("get_schema")
workflow.add_edge("get_schema", "resolve_symbol")
workflow.add_edge("resolve_symbol", "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")
workflow.add_edge("execute_sql", "format_answer")
workflow.add_edge("format_answer", END)
workflow.add_conditional_edges(
    "validate_sql",
    lambda s: "execute_sql" if s['validation']['is_valid'] else END
)
workflow.add_edge("execute_sql", END)


app = workflow.compile()

def ask_agent(question: str):
    """
    Retourne un dict contenant au minimum:
    - sql_query
    - result (raw)
    - answer (formatée)
    - validation
    """
    out = app.invoke({"input": question})

    return out

result = app.invoke({"input": "Prix NVDA hier ?"})
print("SQL généré:", result.get("sql_query", "No SQL"))
print("Réponse:", result.get("answer", "No answer"))
print("Résultat brut:", result.get("result", "No raw result"))