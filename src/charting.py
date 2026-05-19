from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def prepare_price_frame(market_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = market_df[market_df["symbol"] == symbol].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["ma5"] = df["close"].rolling(5, min_periods=1).mean()
    df["ma10"] = df["close"].rolling(10, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = (df["macd_dif"] - df["macd_dea"]) * 2
    if "amount" not in df.columns or df["amount"].fillna(0).sum() == 0:
        df["amount"] = ((df["open"] + df["close"]) / 2 * df["volume"]).astype(float)
        df["amount_estimated"] = True
    else:
        df["amount"] = df["amount"].astype(float)
        df["amount_estimated"] = False
    return df


def generate_technical_charts(market_df: pd.DataFrame, symbol: str, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_price_frame(market_df, symbol)
    if df.empty or len(df) < 35:
        return {}

    chart_paths = {
        "kline_ma": output_dir / f"{symbol.replace('.', '_')}_kline_ma.png",
        "macd": output_dir / f"{symbol.replace('.', '_')}_macd.png",
        "amount": output_dir / f"{symbol.replace('.', '_')}_amount.png",
    }
    _plot_kline_ma(df, symbol, chart_paths["kline_ma"])
    _plot_macd(df, symbol, chart_paths["macd"])
    _plot_amount(df, symbol, chart_paths["amount"])
    return chart_paths


def _setup_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))


def _plot_kline_ma(df: pd.DataFrame, symbol: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    dates = mdates.date2num(df["date"])
    width = 0.6
    for date_num, row in zip(dates, df.itertuples(index=False)):
        color = "#d62728" if row.close >= row.open else "#2ca02c"
        ax.vlines(date_num, row.low, row.high, color=color, linewidth=1)
        body_low = min(row.open, row.close)
        body_height = abs(row.close - row.open) or 0.01
        rect = plt.Rectangle(
            (date_num - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rect)

    ax.plot(df["date"], df["ma5"], label="MA5", color="#1f77b4", linewidth=1.5)
    ax.plot(df["date"], df["ma10"], label="MA10", color="#ff7f0e", linewidth=1.5)
    ax.plot(df["date"], df["ma20"], label="MA20", color="#9467bd", linewidth=1.5)
    ax.set_title(f"{symbol} K-line with MA5/MA10/MA20")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    _setup_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_macd(df: pd.DataFrame, symbol: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    colors = ["#d62728" if value >= 0 else "#2ca02c" for value in df["macd_bar"]]
    ax.bar(df["date"], df["macd_bar"], color=colors, width=0.8, alpha=0.75, label="MACD Bar")
    ax.plot(df["date"], df["macd_dif"], color="#1f77b4", linewidth=1.5, label="DIF")
    ax.plot(df["date"], df["macd_dea"], color="#ff7f0e", linewidth=1.5, label="DEA")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{symbol} MACD")
    ax.set_ylabel("MACD")
    ax.legend(loc="best")
    _setup_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_amount(df: pd.DataFrame, symbol: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    colors = ["#d62728" if row.close >= row.open else "#2ca02c" for row in df.itertuples(index=False)]
    ax.bar(df["date"], df["amount"] / 100000000, color=colors, width=0.8, alpha=0.8, label="Amount")
    title_suffix = " (estimated)" if bool(df["amount_estimated"].iloc[-1]) else ""
    ax.set_title(f"{symbol} Trading Amount{title_suffix}")
    ax.set_ylabel("Amount (100M CNY)")
    ax.legend(loc="best")
    _setup_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
