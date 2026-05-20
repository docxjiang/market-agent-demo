from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PERIOD_WINDOWS = [
    ("1日", 1),
    ("1周", 5),
    ("1个月", 22),
    ("3个月", 66),
    ("6个月", 126),
    ("1年", 252),
    ("最大值", None),
]


def prepare_interactive_price_frame(market_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = market_df[market_df["symbol"] == symbol].copy()
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ["open", "close", "high", "low", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = 0.0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if df["amount"].sum() == 0:
        df["amount"] = ((df["open"] + df["close"]) / 2 * df["volume"]).astype(float)

    close = df["close"].astype(float)
    df["ma5"] = close.rolling(5, min_periods=1).mean()
    df["ma10"] = close.rolling(10, min_periods=1).mean()
    df["ma20"] = close.rolling(20, min_periods=1).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = (df["macd_dif"] - df["macd_dea"]) * 2
    df["rsi14"] = _rsi(close, 14)
    return df


def build_interactive_technical_figure(frame: pd.DataFrame, symbol: str, name: str = "") -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.18, 0.20],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    title = f"{name} {symbol}".strip()
    increasing_color = "#d62728"
    decreasing_color = "#149954"
    volume_colors = [
        increasing_color if row.close >= row.open else decreasing_color
        for row in frame.itertuples(index=False)
    ]

    fig.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="K线",
            increasing_line_color=increasing_color,
            increasing_fillcolor=increasing_color,
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=decreasing_color,
        ),
        row=1,
        col=1,
    )
    for column, color in [("ma5", "#1f77b4"), ("ma10", "#ff7f0e"), ("ma20", "#7f3fbf")]:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                mode="lines",
                line={"width": 1.3, "color": color},
                name=column.upper(),
                hovertemplate=f"{column.upper()}: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["volume"],
            marker_color=volume_colors,
            name="成交量",
            hovertemplate="成交量: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    macd_colors = [increasing_color if value >= 0 else decreasing_color for value in frame["macd_bar"]]
    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["macd_bar"],
            marker_color=macd_colors,
            name="MACD柱",
            hovertemplate="MACD柱: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["macd_dif"],
            mode="lines",
            line={"width": 1.2, "color": "#1f77b4"},
            name="DIF",
            hovertemplate="DIF: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["macd_dea"],
            mode="lines",
            line={"width": 1.2, "color": "#ff7f0e"},
            name="DEA",
            hovertemplate="DEA: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    latest_close = float(frame.iloc[-1]["close"])
    fig.add_hline(y=latest_close, line_dash="dot", line_color="#555555", row=1, col=1)
    fig.update_layout(
        title=title,
        height=720,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        template="plotly_white",
    )
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 7, "label": "1周", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1月", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3月", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6月", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1年", "step": "year", "stepmode": "backward"},
                {"label": "全部", "step": "all"},
            ]
        },
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def build_period_return_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["区间", "涨跌幅", "涨跌幅数值"])
    close = frame["close"].astype(float).reset_index(drop=True)
    latest_close = float(close.iloc[-1])
    rows = []
    for label, rows_back in PERIOD_WINDOWS:
        if rows_back is None:
            start_index = 0
        else:
            start_index = max(0, len(close) - rows_back - 1)
        start_close = float(close.iloc[start_index])
        change = (latest_close / start_close - 1) * 100 if start_close else 0.0
        rows.append({"区间": label, "涨跌幅": f"{change:.2f}%", "涨跌幅数值": change})
    return pd.DataFrame(rows)


def build_key_metrics_table(frame: pd.DataFrame, quote: dict[str, Any] | None = None) -> pd.DataFrame:
    quote = quote or {}
    latest = frame.iloc[-1] if not frame.empty else {}
    latest_price = _number_or_quote(latest, quote, "close", "latest_price")
    open_price = _number_or_quote(latest, quote, "open", "quote_open")
    prev_close = _safe_float(quote.get("prev_close"))
    high = _number_or_quote(latest, quote, "high", "quote_high")
    low = _number_or_quote(latest, quote, "low", "quote_low")
    volume = _safe_float(quote.get("volume")) or _safe_float(latest.get("volume") if hasattr(latest, "get") else 0)
    amount = _safe_float(quote.get("amount")) or _safe_float(latest.get("amount") if hasattr(latest, "get") else 0)
    rsi14 = _safe_float(latest.get("rsi14") if hasattr(latest, "get") else 0)

    rows = [
        ("最新价", _fmt_price(latest_price)),
        ("昨收", _fmt_price(prev_close)),
        ("开盘", _fmt_price(open_price)),
        ("当日幅度", f"{_fmt_price(low)} - {_fmt_price(high)}" if low and high else "-"),
        ("量", _fmt_compact(volume)),
        ("成交额", _fmt_compact(amount)),
        ("市值", _fmt_compact(_safe_float(quote.get("market_cap")))),
        ("动态PE", _fmt_optional(_safe_float(quote.get("pe_dynamic")))),
        ("市盈率", _fmt_optional(_safe_float(quote.get("pe_dynamic")))),
        ("市净率", _fmt_optional(_safe_float(quote.get("pb")))),
        ("换手率", _fmt_percent(_safe_float(quote.get("turnover_rate_pct")))),
        ("RSI(14)", _fmt_optional(rsi14)),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值"])


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, pd.NA)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _number_or_quote(row: Any, quote: dict[str, Any], row_key: str, quote_key: str) -> float:
    quoted = _safe_float(quote.get(quote_key))
    if quoted:
        return quoted
    if hasattr(row, "get"):
        return _safe_float(row.get(row_key))
    return 0.0


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_price(value: float) -> str:
    return f"{value:.2f}" if value else "-"


def _fmt_optional(value: float) -> str:
    return f"{value:.2f}" if value else "-"


def _fmt_percent(value: float) -> str:
    return f"{value:.2f}%" if value else "-"


def _fmt_compact(value: float) -> str:
    if not value:
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"
