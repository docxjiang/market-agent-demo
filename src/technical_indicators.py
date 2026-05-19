from __future__ import annotations

from typing import Any

import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def latest(series: pd.Series, default: float = 0.0) -> float:
    clean = series.dropna()
    if clean.empty:
        return default
    return float(clean.iloc[-1])


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)


def calculate_indicators(group: pd.DataFrame, quote: dict[str, Any] | None = None) -> dict[str, Any]:
    quote = quote or {}
    df = group.sort_values("date").copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = (dif - dea) * 2

    low9 = low.rolling(9, min_periods=1).min()
    high9 = high.rolling(9, min_periods=1).max()
    rsv = (close - low9) / (high9 - low9).replace(0, pd.NA) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    ma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=2).std()
    boll_upper = ma20 + 2 * std20
    boll_lower = ma20 - 2 * std20
    boll_width = (boll_upper - boll_lower) / ma20.replace(0, pd.NA) * 100
    boll_percent_b = (close - boll_lower) / (boll_upper - boll_lower).replace(0, pd.NA) * 100

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=1).mean()
    daily_return = close.pct_change()
    volatility20 = daily_return.rolling(20, min_periods=2).std() * (252 ** 0.5) * 100

    rolling_peak = close.cummax()
    drawdown = close / rolling_peak - 1

    latest_close = latest(close)
    high_max = float(high.max())
    low_min = float(low.min())
    close_position = (latest_close - low_min) / (high_max - low_min) * 100 if high_max > low_min else 50.0
    period_return = (latest_close / float(close.iloc[0]) - 1) * 100 if float(close.iloc[0]) else 0.0
    volume_ma5 = volume.rolling(5, min_periods=1).mean()
    volume_ratio_5 = latest(volume) / latest(volume_ma5, latest(volume)) if latest(volume_ma5, 0) else 1.0

    indicators: dict[str, Any] = {
        "start_date": str(df.iloc[0]["date"]),
        "end_date": str(df.iloc[-1]["date"]),
        "open": latest(open_),
        "close": latest_close,
        "high": high_max,
        "low": low_min,
        "period_return_pct": period_return,
        "amplitude_pct": (high_max / low_min - 1) * 100 if low_min else 0.0,
        "max_drawdown_pct": latest(drawdown.cummin() * 100),
        "ma5": latest(close.rolling(5, min_periods=1).mean()),
        "ma10": latest(close.rolling(10, min_periods=1).mean()),
        "ma20": latest(ma20),
        "ma60": latest(close.rolling(60, min_periods=1).mean()),
        "ema12": latest(ema12),
        "ema26": latest(ema26),
        "macd_dif": latest(dif),
        "macd_dea": latest(dea),
        "macd_bar": latest(macd_bar),
        "rsi6": latest(rsi(close, 6), 50.0),
        "rsi12": latest(rsi(close, 12), 50.0),
        "rsi24": latest(rsi(close, 24), 50.0),
        "kdj_k": latest(k, 50.0),
        "kdj_d": latest(d, 50.0),
        "kdj_j": latest(j, 50.0),
        "boll_mid": latest(ma20),
        "boll_upper": latest(boll_upper),
        "boll_lower": latest(boll_lower),
        "boll_width_pct": latest(boll_width),
        "boll_percent_b": latest(boll_percent_b, 50.0),
        "atr14": latest(atr14),
        "volatility20_pct": latest(volatility20),
        "volume": latest(volume),
        "volume_ratio_5": volume_ratio_5,
        "close_position_pct": close_position,
        "turnover_rate_pct": safe_float(quote.get("turnover_rate_pct")),
        "pe_dynamic": safe_float(quote.get("pe_dynamic")),
        "pb": safe_float(quote.get("pb")),
        "market_cap": safe_float(quote.get("market_cap")),
        "circulating_market_cap": safe_float(quote.get("circulating_market_cap")),
        "quote_pct_change": safe_float(quote.get("pct_change")),
    }
    return indicators


def classify_indicators(indicators: dict[str, Any]) -> dict[str, str]:
    close = indicators["close"]
    ma5 = indicators["ma5"]
    ma20 = indicators["ma20"]
    ma60 = indicators["ma60"]
    macd_bar = indicators["macd_bar"]
    rsi6_value = indicators["rsi6"]
    rsi24_value = indicators["rsi24"]
    kdj_j = indicators["kdj_j"]
    percent_b = indicators["boll_percent_b"]
    volume_ratio = indicators["volume_ratio_5"]
    pe = indicators["pe_dynamic"]
    pb = indicators["pb"]

    trend = "多头占优" if close > ma5 > ma20 > ma60 else "空头或震荡" if close < ma5 < ma20 else "结构分化"
    macd = "动能改善" if indicators["macd_dif"] > indicators["macd_dea"] and macd_bar > 0 else "动能偏弱"
    rsi_state = "短线偏热" if rsi6_value >= 70 else "短线偏冷" if rsi6_value <= 30 else "短线中性"
    medium_rsi = "中期强势" if rsi24_value >= 55 else "中期弱势" if rsi24_value <= 45 else "中期中性"
    kdj_state = "超买风险" if kdj_j >= 100 else "超卖修复" if kdj_j <= 0 else "摆动区间内"
    boll_state = "贴近上轨" if percent_b >= 80 else "贴近下轨" if percent_b <= 20 else "布林中轨附近"
    volume_state = "放量" if volume_ratio >= 1.3 else "缩量" if volume_ratio <= 0.75 else "量能平稳"
    if pe <= 0 and pb <= 0:
        valuation_state = "估值数据缺失"
    else:
        valuation_state = "估值偏高" if pe >= 30 or pb >= 8 else "估值中性偏高" if pe >= 20 or pb >= 5 else "估值相对不高"
    return {
        "trend": trend,
        "macd": macd,
        "rsi": rsi_state,
        "medium_rsi": medium_rsi,
        "kdj": kdj_state,
        "boll": boll_state,
        "volume": volume_state,
        "valuation": valuation_state,
    }


def format_indicator_table(indicators: dict[str, Any]) -> list[str]:
    rows = [
        ("收盘价", f"{indicators['close']:.2f}", "最新价格观察基准"),
        ("区间收益率", f"{indicators['period_return_pct']:.2f}%", "衡量样本期方向"),
        ("区间振幅", f"{indicators['amplitude_pct']:.2f}%", "衡量高低点波动空间"),
        ("最大回撤", f"{indicators['max_drawdown_pct']:.2f}%", "衡量样本期下行压力"),
        ("MA5", f"{indicators['ma5']:.2f}", "短线均线"),
        ("MA10", f"{indicators['ma10']:.2f}", "短中线均线"),
        ("MA20", f"{indicators['ma20']:.2f}", "月度均线"),
        ("MA60", f"{indicators['ma60']:.2f}", "季度均线"),
        ("EMA12", f"{indicators['ema12']:.2f}", "MACD 快线基础"),
        ("EMA26", f"{indicators['ema26']:.2f}", "MACD 慢线基础"),
        ("MACD DIF", f"{indicators['macd_dif']:.2f}", "快慢均线差"),
        ("MACD DEA", f"{indicators['macd_dea']:.2f}", "DIF 平滑线"),
        ("MACD 柱", f"{indicators['macd_bar']:.2f}", "动能变化"),
        ("RSI6", f"{indicators['rsi6']:.2f}", "短线强弱"),
        ("RSI12", f"{indicators['rsi12']:.2f}", "中短线强弱"),
        ("RSI24", f"{indicators['rsi24']:.2f}", "中期强弱"),
        ("KDJ K", f"{indicators['kdj_k']:.2f}", "随机指标 K 值"),
        ("KDJ D", f"{indicators['kdj_d']:.2f}", "随机指标 D 值"),
        ("KDJ J", f"{indicators['kdj_j']:.2f}", "敏感摆动值"),
        ("BOLL 中轨", f"{indicators['boll_mid']:.2f}", "20 日均线"),
        ("BOLL 上轨", f"{indicators['boll_upper']:.2f}", "压力观察位"),
        ("BOLL 下轨", f"{indicators['boll_lower']:.2f}", "支撑观察位"),
        ("BOLL 带宽", f"{indicators['boll_width_pct']:.2f}%", "波动收敛/扩张"),
        ("BOLL %B", f"{indicators['boll_percent_b']:.2f}", "价格在布林区间的位置"),
        ("ATR14", f"{indicators['atr14']:.2f}", "平均真实波幅"),
        ("20日年化波动率", f"{indicators['volatility20_pct']:.2f}%", "近期波动水平"),
        ("成交量/5日均量", f"{indicators['volume_ratio_5']:.2f}", "量能确认"),
        ("区间价格分位", f"{indicators['close_position_pct']:.2f}%", "收盘价在样本高低点中的位置"),
        ("换手率", fmt_optional(indicators["turnover_rate_pct"], suffix="%"), "交易活跃度"),
        ("动态PE", fmt_optional(indicators["pe_dynamic"]), "估值指标"),
        ("PB", fmt_optional(indicators["pb"]), "净资产估值"),
    ]
    return [f"| {name} | {value} | {meaning} |" for name, value, meaning in rows]


def fmt_optional(value: float, suffix: str = "") -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"{value:.2f}{suffix}"
