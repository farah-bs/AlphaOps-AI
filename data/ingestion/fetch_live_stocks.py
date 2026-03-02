import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}")
SCHEMA = 'public'

tickers = {
    'SPY': 'SP500_ETF',
    'QQQ': 'Nasdaq100_ETF',
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'BTC-USD': 'Bitcoin'
}

def ensure_year_partition(engine, year: int):
    start = f"{year}-01-01"
    end = f"{year+1}-01-01"
    sql = f"""
    CREATE TABLE IF NOT EXISTS fact_ohlcv_{year}
    PARTITION OF fact_ohlcv
    FOR VALUES FROM ('{start}') TO ('{end}');
    """
    with engine.begin() as conn:
        conn.execute(text(sql))

def get_last_loaded_date(engine, symbol: str) -> date | None:
    q = text('SELECT MAX(date) FROM fact_ohlcv WHERE symbol = :symbol')
    with engine.begin() as conn:
        return conn.execute(q, {"symbol": symbol}).scalar()

def upsert_dim_tickers(engine, df: pd.DataFrame):
    table = 'dim_tickers'
    rows = df.to_dict(orient="records")

    tbl = pd.io.sql.SQLTable(table, pd.io.sql.pandasSQL_builder(engine), frame=df, index=False).table
    stmt = insert(tbl).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=['symbol'],
        set_={
            'name': stmt.excluded.name,
            'market': stmt.excluded.market,
            'first_date': stmt.excluded.first_date,
            'last_date': stmt.excluded.last_date,
            'avg_volume': stmt.excluded.avg_volume,
        }
    )

    with engine.begin() as conn:
        conn.execute(stmt)


def upsert_fact_ohlcv(engine, df: pd.DataFrame):
    table = 'fact_ohlcv'
    rows = df.to_dict(orient="records")

    tbl = pd.io.sql.SQLTable(table, pd.io.sql.pandasSQL_builder(engine), frame=df, index=False).table
    stmt = insert(tbl).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=['symbol', 'date'],
        set_={
            'open_price': stmt.excluded.open_price,
            'high_price': stmt.excluded.high_price,
            'low_price': stmt.excluded.low_price,
            'close_price': stmt.excluded.close_price,
            'volume': stmt.excluded.volume,
            'adj_close': stmt.excluded.adj_close,
        }
    )

    with engine.begin() as conn:
        conn.execute(stmt)

def fetch_incremental(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        yf_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        yf_start = start_date.strftime("%Y-%m-%d")

        hist = yf.download(symbol, start=yf_start, end=yf_end, auto_adjust=False, progress=False)
        if hist is None or hist.empty:
            return pd.DataFrame()

        df = hist.reset_index()
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        df['date'] = pd.to_datetime(df[date_col]).dt.date
        df['symbol'] = symbol

        fact_df = df[['date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        fact_df.columns = ['date', 'symbol', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        fact_df['adj_close'] = fact_df['close_price']
        fact_df = fact_df.dropna()

        return fact_df
    except Exception as e:
        print(f"{symbol}; fetch error: {str(e)}")
        return pd.DataFrame()

def run_daily_batch():
    today = datetime.now().date()
    ensure_year_partition(engine, today.year)

    for symbol, name in tickers.items():
        last_date = get_last_loaded_date(engine, symbol)

        if last_date is None:
            start = date(2020, 1, 1)
        else:
            start = last_date + timedelta(days=1)

        if start > today:
            print(f"{symbol}: up-to-date (last_date={last_date})")
            continue

        print(f"{symbol}: fetching {start} -> {today}")
        fact_df = fetch_incremental(symbol, start, today)

        if fact_df.empty or fact_df is None:
            print(f"{symbol}: no new rows")
            continue

        ticker_info = pd.DataFrame([{
            'symbol': symbol,
            'name': name,
            'market': name if 'ETF' in name else ('Crypto' if 'BTC' in symbol else 'Equity'),
            'first_date': fact_df['date'].min() if last_date is None else None,
            'last_date': fact_df['date'].max(),
            'avg_volume': int(fact_df['volume'].mean())
        }])

        if ticker_info.loc[0, 'first_date'] is None:
            with engine.begin() as conn:
                existing_first = conn.execute(
                    text("SELECT first_date FROM dim_tickers WHERE symbol=:symbol"),
                    {"symbol": symbol}
                ).scalar()
            ticker_info.loc[0, 'first_date'] = existing_first
        
        upsert_dim_tickers(engine, ticker_info)

        upsert_fact_ohlcv(engine, fact_df)

        

        print(f"{symbol}: inserted/updated {len(fact_df)} rows. last={fact_df['date'].max()}")

    stats = pd.read_sql(
        'SELECT symbol, COUNT(*) as count, MIN(date) as first_date, MAX(date) as last_date '
        'FROM fact_ohlcv GROUP BY symbol ORDER BY symbol',
        engine
    )
    print(stats)

if __name__ == "__main__":
    run_daily_batch()