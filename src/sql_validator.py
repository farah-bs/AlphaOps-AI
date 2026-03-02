import sqlglot
from sqlglot import parse_one, exp
from typing import Dict, Any, List
import re

class SQLValidator:
    def __init__(
        self,
        allowed_tables: List[str] = None,
        max_limit: int = 500,
        require_limit_on_fact: bool = True,
    ):
        self.allowed_tables = allowed_tables or ["dim_tickers", "dimtime", "fact_ohlcv"]
        self.max_limit = max_limit
        self.require_limit_on_fact = require_limit_on_fact

    def _is_select_like(self, parsed) -> bool:
        # SELECT ...
        if isinstance(parsed, exp.Select):
            return True
        # WITH ... SELECT ...
        if isinstance(parsed, exp.With):
            return isinstance(parsed.this, exp.Select)
        # Parfois parse_one renvoie Statement -> .this
        if hasattr(parsed, "this"):
            return self._is_select_like(parsed.this)
        return False

    def validate(self, query: str) -> Dict[str, Any]:
        try:
            q = query.strip()
            parsed = parse_one(q, dialect="postgres")

            # 1) uniquement SELECT / WITH..SELECT
            if not self._is_select_like(parsed):
                return {"is_valid": False, "reason": "Seulement SELECT autorisé (WITH...SELECT ok)."}
            
            # 2) tables autorisées
            cte_names = set()

            # Trouver le nœud WITH, peu importe que parsed soit Select/With/Statement
            with_node = next(parsed.find_all(exp.With), None)
            if with_node:
                # with_node.expressions contient les CTE
                for cte in with_node.expressions:
                    # cte est souvent un exp.CTE
                    # alias_or_name marche bien pour récupérer le nom du CTE
                    cte_names.add(cte.alias_or_name)

            tables = list(parsed.find_all(exp.Table))

            print("CTE NAMES:", cte_names)
            print("TABLES FOUND:", [t.name for t in tables])

            invalid = []
            for t in tables:
                name = t.name
                # si c'est un CTE, OK
                if name in cte_names:
                    continue
                # sinon whitelist stricte
                if name not in self.allowed_tables:
                    invalid.append(name)

            if invalid:
                return {"is_valid": False, "reason": f"Tables interdites: {invalid}"}

            # 3) patterns dangereux (en upper)
            upper = q.upper()
            dangerous = [";--", "/*", "*/", "UNION", "PG_SLEEP", "DROP", "ALTER", "TRUNCATE", "INSERT", "UPDATE", "DELETE"]
            if any(pat in upper for pat in dangerous):
                return {"is_valid": False, "reason": "Pattern potentiellement dangereux détecté."}

            # 4) limiter les dumps sur fact_ohlcv
            uses_fact = any(t.name == "fact_ohlcv" for t in tables)

            # LIMIT check
            limit_exp = parsed.args.get("limit")
            if limit_exp is not None:
                lit = limit_exp.this
                if isinstance(lit, exp.Literal) and lit.is_int:
                    lim = int(lit.this)
                    if lim <= 0:
                        return {"is_valid": False, "reason": "LIMIT <= 0 interdit."}
                    if lim > self.max_limit:
                        return {"is_valid": False, "reason": f"LIMIT trop grand (max {self.max_limit})."}
            else:
                if uses_fact and self.require_limit_on_fact:
                    return {"is_valid": False, "reason": f"Requête sur fact_ohlcv sans LIMIT (ajoute LIMIT <= {self.max_limit})."}

            # * check (éviter SELECT * sur fact)
            has_star = any(isinstance(x, exp.Star) for x in parsed.find_all(exp.Star))
            if uses_fact and has_star:
                return {"is_valid": False, "reason": "SELECT * interdit sur fact_ohlcv (sélectionne les colonnes nécessaires)."}

            return {"is_valid": True, "reason": "Query safe", "validated_sql": q}

        except Exception as e:
            return {"is_valid": False, "reason": f"Erreur parse/validation: {str(e)}"}