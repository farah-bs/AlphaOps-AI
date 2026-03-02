-- Star schema initialization
-- Executed once at first container boot (volume empty)

-- DimTickers
CREATE TABLE IF NOT EXISTS dim_tickers (
    symbol      VARCHAR(20) PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    market      VARCHAR(20) NOT NULL,
    sector      VARCHAR(50),
    first_date  DATE,
    last_date   DATE,
    avg_volume  BIGINT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DimTime
CREATE TABLE IF NOT EXISTS dimtime (
    date        DATE PRIMARY KEY,
    year        INT NOT NULL,
    month       INT NOT NULL,
    day         INT NOT NULL,
    quarter     INT NOT NULL,
    day_of_week INT NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    is_month_end    BOOLEAN NOT NULL
);

-- Seed dimtime 2020–2026
INSERT INTO dimtime
SELECT
    d::date,
    EXTRACT(YEAR    FROM d)::INT,
    EXTRACT(MONTH   FROM d)::INT,
    EXTRACT(DAY     FROM d)::INT,
    EXTRACT(QUARTER FROM d)::INT,
    EXTRACT(ISODOW  FROM d)::INT - 1,
    EXTRACT(ISODOW  FROM d) IN (6, 7),
    (d::date = (date_trunc('month', d::date) + interval '1 month - 1 day')::date)
FROM generate_series('2020-01-01'::date, '2026-12-31'::date, '1 day'::interval) d
ON CONFLICT (date) DO NOTHING;

-- FactOHLCV (partitioned by year)
CREATE TABLE IF NOT EXISTS fact_ohlcv (
    symbol      VARCHAR(20) NOT NULL REFERENCES dim_tickers(symbol),
    date        DATE        NOT NULL REFERENCES dimtime(date),
    open_price  DECIMAL(12,4) NOT NULL,
    high_price  DECIMAL(12,4) NOT NULL,
    low_price   DECIMAL(12,4) NOT NULL,
    close_price DECIMAL(12,4) NOT NULL,
    volume      BIGINT NOT NULL,
    adj_close   DECIMAL(12,4),
    volatility  DECIMAL(8,4) GENERATED ALWAYS AS ((high_price - low_price) / close_price) STORED,
    PRIMARY KEY (symbol, date),
    CHECK (open_price >= 0),
    CHECK (volume >= 0)
) PARTITION BY RANGE (date);

CREATE TABLE IF NOT EXISTS fact_ohlcv_2020 PARTITION OF fact_ohlcv FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
CREATE TABLE IF NOT EXISTS fact_ohlcv_2021 PARTITION OF fact_ohlcv FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
CREATE TABLE IF NOT EXISTS fact_ohlcv_2022 PARTITION OF fact_ohlcv FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
CREATE TABLE IF NOT EXISTS fact_ohlcv_2023 PARTITION OF fact_ohlcv FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
CREATE TABLE IF NOT EXISTS fact_ohlcv_2024 PARTITION OF fact_ohlcv FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE IF NOT EXISTS fact_ohlcv_2025 PARTITION OF fact_ohlcv FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS fact_ohlcv_2026 PARTITION OF fact_ohlcv FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Indexes pour NL2SQL
CREATE INDEX IF NOT EXISTS idx_fact_symbol_date ON fact_ohlcv (symbol, date);
CREATE INDEX IF NOT EXISTS idx_fact_date_symbol ON fact_ohlcv (date, symbol);
CREATE INDEX IF NOT EXISTS idx_fact_volume      ON fact_ohlcv (volume DESC);
CREATE INDEX IF NOT EXISTS idx_fact_close       ON fact_ohlcv (close_price DESC);
