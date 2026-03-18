import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
import ta
from src.db.connection import get_engine

FEATURE_COLS = [
    "log_return",
    "return_mean_5", "return_std_5",
    "return_mean_10", "return_std_10",
    "return_mean_20", "return_std_20",
    "rsi_14",
    "macd_pct", "macd_signal_pct",
    "volume_log_change",
    "volatility",
    "bb_pct",
    "atr_pct",
    "sma5_vs_sma20",
    "spy_return",
]

# Tickers qui sont eux-mêmes le marché → spy_return = 0
_MARKET_TICKERS = {"SPY", "QQQ"}


def fetch_ohlcv(ticker: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Fetch OHLCV rows for `ticker` from fact_ohlcv, ordered by date."""
    engine = get_engine()
    query = text("""
        SELECT date, open_price, high_price, low_price,
               close_price, adj_close, volume, volatility
        FROM fact_ohlcv
        WHERE symbol = :symbol
          AND date >= :start_date
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": ticker, "start_date": start_date})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to the OHLCV dataframe."""
    # Fallback : utiliser close_price si adj_close est NaN (ex: crypto sans adj)
    if "close_price" in df.columns:
        df = df.copy()
        df["adj_close"] = df["adj_close"].fillna(df["close_price"])

    close = df["adj_close"]

    df["log_return"] = np.log(close / close.shift(1))

    for w in [5, 10, 20]:
        df[f"return_mean_{w}"] = df["log_return"].rolling(w).mean()
        df[f"return_std_{w}"] = df["log_return"].rolling(w).std()

    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_ind = ta.trend.MACD(close=close)
    # Normalize by price so the feature is scale-invariant across different price levels
    # and time periods (prevents distribution shift when stock price changes significantly)
    df["macd_pct"]        = macd_ind.macd()        / close
    df["macd_signal_pct"] = macd_ind.macd_signal() / close

    df["volume_log_change"] = np.log(df["volume"] / df["volume"].shift(1))

    # volatility column already present from fact_ohlcv

    # Bollinger Bands — position du prix dans la bande (0=bas, 1=haut)
    bb = ta.volatility.BollingerBands(close=close, window=20)
    df["bb_pct"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)

    # ATR normalisé par le prix
    df["atr_pct"] = ta.volatility.AverageTrueRange(
        high=df["high_price"], low=df["low_price"], close=close, window=14
    ).average_true_range() / close

    # Crossover SMA5 vs SMA20
    df["sma5_vs_sma20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1

    df = df.dropna(subset=FEATURE_COLS)
    return df


def _build_sequences(
    scaled_X: np.ndarray,
    raw_y: np.ndarray,
    seq_len: int,
):
    """Sliding-window: input = seq_len scaled feature rows, target = next-day raw log return."""
    X_list, y_list = [], []
    for i in range(seq_len, len(scaled_X)):
        X_list.append(scaled_X[i - seq_len : i])
        y_list.append(raw_y[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def prepare_data(
    ticker: str,
    seq_len: int = 60,
    train_end: str = "2023-12-31",
    val_end: str = "2024-06-30",
):
    """
    Full pipeline for one ticker:
      fetch → features → time-split → fit scaler on train → build sequences

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
    """
    df = fetch_ohlcv(ticker)
    df = compute_features(df)

    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]

    if len(train_df) < seq_len + 1:
        raise ValueError(
            f"Not enough training rows for {ticker} "
            f"(need > {seq_len}, got {len(train_df)})"
        )

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)

    X_train = scaler.transform(train_df[FEATURE_COLS].values)
    X_val = scaler.transform(val_df[FEATURE_COLS].values)
    X_test = scaler.transform(test_df[FEATURE_COLS].values)

    y_train_raw = train_df["log_return"].values
    y_val_raw = val_df["log_return"].values
    y_test_raw = test_df["log_return"].values

    X_train_seq, y_train = _build_sequences(X_train, y_train_raw, seq_len)
    X_val_seq, y_val = _build_sequences(X_val, y_val_raw, seq_len)
    X_test_seq, y_test = _build_sequences(X_test, y_test_raw, seq_len)

    return (X_train_seq, y_train), (X_val_seq, y_val), (X_test_seq, y_test), scaler


def get_last_sequence(ticker: str, seq_len: int, scaler: StandardScaler) -> np.ndarray:
    """
    Fetch the most recent `seq_len` rows and return a scaled feature tensor
    ready for inference. Shape: (1, seq_len, n_features).
    """
    df = fetch_ohlcv(ticker)
    df = compute_features(df)
    if len(df) < seq_len:
        raise ValueError(f"Not enough rows for {ticker} (need {seq_len}, got {len(df)})")
    recent = df.tail(seq_len)[FEATURE_COLS].values
    scaled = scaler.transform(recent)
    return scaled[np.newaxis, :, :].astype(np.float32)  # (1, seq_len, n_features)


def _build_sequences_lstm(
    scaled_X:   np.ndarray,
    raw_prices: np.ndarray,
    dates:      "pd.DatetimeIndex",
    seq_len:    int,
    horizons:   tuple = (1, 7, 30),
) -> tuple:
    """
    Construit des séquences (X, y) pour le LSTM multi-horizon sans data leakage.

    Pour chaque indice i dans [seq_len, N - max_horizon] :
        X[i] = scaled_X[i-seq_len : i]          → features historiques
        ref   = raw_prices[i-1]                  → dernier prix connu
        y_h   = int(raw_prices[i + h - 1] > ref) → direction à horizon h

    La date associée à la séquence est dates[i-1] (dernier jour dans la fenêtre).
    Ceci permet de splitter train/val/test par date sans leakage sur les features.

    Returns:
        X_arr      : np.ndarray (N_seq, seq_len, n_features)  float32
        y_arr      : np.ndarray (N_seq, len(horizons))         float32
        date_index : pd.DatetimeIndex des derniers jours de chaque fenêtre
    """
    max_h   = max(horizons)
    X_list, y_list, date_list = [], [], []

    for i in range(seq_len, len(scaled_X) - max_h + 1):
        X_list.append(scaled_X[i - seq_len : i])
        ref = raw_prices[i - 1]
        # J+1 : lissé sur 3 jours pour réduire le bruit du signal quotidien
        # J+7 et J+30 : direction classique (déjà stables)
        targets = [
            int(np.mean(raw_prices[i:i+3]) > ref) if h == 1
            else int(raw_prices[i + h - 1] > ref)
            for h in horizons
        ]
        y_list.append(targets)
        date_list.append(dates[i - 1])

    return (
        np.array(X_list,  dtype=np.float32),
        np.array(y_list,  dtype=np.float32),
        pd.DatetimeIndex(date_list),
    )


def prepare_data_lstm(
    ticker:    str,
    seq_len:   int   = 60,
    train_end: str   = "2023-12-31",
    val_end:   str   = "2024-06-30",
    horizons:  tuple = (1, 7, 30),
):
    """
    Pipeline complet LSTM multi-horizon pour un ticker.

    Garanties anti-leakage :
        - Le scaler est fit UNIQUEMENT sur les features de train
        - Les targets sont calculés depuis raw adj_close (non scalé)
        - Le split train/val/test se fait par date du dernier jour de la fenêtre

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
        y shape : (N, len(horizons))  — colonnes = [J+1, J+7, J+30]
    """
    df = fetch_ohlcv(ticker)
    df = compute_features(df)

    # ── spy_return : contexte macro du marché ─────────────────────────────────
    if ticker.upper() in _MARKET_TICKERS:
        df["spy_return"] = 0.0
    else:
        try:
            spy_df = fetch_ohlcv("SPY")
            spy_df = compute_features(spy_df)
            df["spy_return"] = spy_df["log_return"].reindex(df.index).fillna(0.0)
        except Exception:
            df["spy_return"] = 0.0

    max_h     = max(horizons)
    train_df  = df[df.index <= train_end]

    if len(train_df) < seq_len + max_h + 1:
        raise ValueError(
            f"Pas assez de données pour {ticker} "
            f"(besoin > {seq_len + max_h}, got {len(train_df)})"
        )

    # Scaler fit sur le train uniquement
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)

    # Transformation de TOUTES les features (train + val + test)
    X_all      = scaler.transform(df[FEATURE_COLS].values)
    prices_all = df["adj_close"].values
    dates_all  = df.index

    # Construction des séquences multi-horizon sur le dataset complet
    X_seq, y_seq, seq_dates = _build_sequences_lstm(
        X_all, prices_all, dates_all, seq_len, horizons
    )

    # Split par date du dernier élément de chaque séquence
    # Gap de seq_len jours entre train et val pour éviter l'overlap de features
    t_end      = pd.Timestamp(train_end)
    val_start  = t_end + pd.Timedelta(days=seq_len)
    v_end      = pd.Timestamp(val_end)

    train_mask = seq_dates <= t_end
    val_mask   = (seq_dates > val_start) & (seq_dates <= v_end)
    test_mask  = seq_dates > v_end

    return (
        (X_seq[train_mask], y_seq[train_mask]),
        (X_seq[val_mask],   y_seq[val_mask]),
        (X_seq[test_mask],  y_seq[test_mask]),
        scaler,
    )


def prepare_prophet_df(ticker: str, window: int) -> pd.DataFrame:
    """
    Retourne les `window` derniers jours du ticker au format Prophet (ds, y)
    avec les vraies dates et les prix bruts (adj_close).
    Utilisé par l'API de serving pour l'inférence.

    Args:
        ticker : ticker cible (ex: "AAPL")
        window : nombre de jours en entrée (60 pour daily, 180 pour monthly)
    """
    df = fetch_ohlcv(ticker)
    if len(df) < window:
        raise ValueError(
            f"Pas assez de données pour {ticker} (besoin {window}, trouvé {len(df)})"
        )
    recent = df.tail(window).copy()
    return pd.DataFrame({
        "ds": recent.index,
        "y":  recent["adj_close"].values,
    }).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), sc = prepare_data(args.ticker)
    print(f"Ticker: {args.ticker}")
    print(f"  Train sequences : {X_tr.shape}  targets: {y_tr.shape}")
    print(f"  Val   sequences : {X_v.shape}  targets: {y_v.shape}")
    print(f"  Test  sequences : {X_te.shape}  targets: {y_te.shape}")
    print(f"  Features        : {len(FEATURE_COLS)} → {FEATURE_COLS}")
