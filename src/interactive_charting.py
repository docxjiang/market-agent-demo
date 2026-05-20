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

CHART_WINDOW_OPTIONS = ["1日", "1周", "1月", "3个月", "6个月", "1年", "全部"]


def prepare_interactive_price_frame(market_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if market_df.empty or "symbol" not in market_df.columns:
        return pd.DataFrame()

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


def select_chart_window_frame(
    daily_frame: pd.DataFrame,
    intraday_frame: pd.DataFrame | None,
    window: str,
) -> pd.DataFrame:
    if daily_frame.empty:
        return daily_frame

    if window in {"1日", "1周"} and intraday_frame is not None and not intraday_frame.empty:
        source = _select_intraday_source(intraday_frame, window)
        latest = source["date"].max()
        start = latest.normalize() if window == "1日" else latest - pd.Timedelta(days=7)
        filtered = source[source["date"] >= start].copy()
        return filtered if not filtered.empty else source

    source = daily_frame.copy()
    latest = source["date"].max()
    if window == "1月":
        start = latest - pd.DateOffset(months=1)
    elif window == "3个月":
        start = latest - pd.DateOffset(months=3)
    elif window == "6个月":
        start = latest - pd.DateOffset(months=6)
    elif window == "1年":
        start = latest - pd.DateOffset(years=1)
    elif window == "1周":
        start = latest - pd.Timedelta(days=7)
    elif window == "1日":
        start = latest.normalize()
    else:
        return source

    filtered = source[source["date"] >= start].copy()
    return filtered if not filtered.empty else source.tail(1).copy()


def format_chart_range(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "-"
    start = pd.to_datetime(frame.iloc[0]["date"], errors="coerce")
    end = pd.to_datetime(frame.iloc[-1]["date"], errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return "-"
    if start.normalize() == start and end.normalize() == end:
        return f"{start:%Y-%m-%d} 至 {end:%Y-%m-%d}"
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M} 至 {end:%H:%M}"
    return f"{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}"


def format_data_source_label(source: str) -> str:
    normalized = str(source or "").lower()
    if normalized.startswith("eastmoney"):
        provider = "东方财富"
    elif normalized.startswith("sina"):
        provider = "新浪财经"
    elif normalized.startswith("akshare"):
        provider = "AKShare"
    else:
        provider = source or "未知"

    if "60m" in normalized:
        return f"{provider}（60分钟）"
    if "30m" in normalized:
        return f"{provider}（30分钟）"
    if "15m" in normalized:
        return f"{provider}（15分钟）"
    if "5m" in normalized:
        return f"{provider}（5分钟）"
    if "1m" in normalized:
        return f"{provider}（1分钟）"
    return f"{provider}（日线）"


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
        title={"text": title, "x": 0.0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        height=720,
        dragmode="pan",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(255,255,255,0.85)",
        },
        template="plotly_white",
        uirevision=f"{symbol}-technical-chart",
    )
    fig.update_xaxes(
        fixedrange=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(fixedrange=False, row=2, col=1)
    fig.update_xaxes(fixedrange=False, row=3, col=1)
    fig.update_yaxes(title_text="价格", fixedrange=True, row=1, col=1)
    fig.update_yaxes(title_text="成交量", fixedrange=True, row=2, col=1)
    fig.update_yaxes(title_text="MACD", fixedrange=True, row=3, col=1)
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
    rs = gain / loss.mask(loss == 0)
    return (100 - 100 / (1 + rs)).fillna(50.0).astype(float)


def _select_intraday_source(intraday_frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if "data_source" not in intraday_frame.columns:
        return intraday_frame.copy()

    source = intraday_frame.copy()
    data_source = source["data_source"].astype(str)
    preferred_patterns = ["60m"] if window == "1周" else ["5m", "1m", "15m", "30m"]
    for pattern in preferred_patterns:
        filtered = source[data_source.str.contains(pattern, na=False)].copy()
        if not filtered.empty:
            return filtered
    return source


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
