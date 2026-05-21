from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm_client import LLMClient
from src.news_evidence import (
    classify_news_item_sentiment,
    normalized_sentiment_label,
    select_representative_news_evidence,
)
from src.technical_indicators import calculate_indicators, classify_indicators, format_indicator_table


DISCLAIMER = "免责声明：本系统仅用于课程演示和市场信息解读，不构成投资建议。"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
logger = logging.getLogger(__name__)
LLM_AGENT_LABELS = {
    "news_agent.txt": "新闻解读智能体",
    "technical_agent.txt": "技术观察智能体",
    "risk_agent.txt": "风险提示智能体",
}


@dataclass
class AgentResult:
    name: str
    content: str


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _try_llm_agent(prompt_name: str, payload: dict[str, Any], llm_client: Any | None = None) -> str:
    agent_label = LLM_AGENT_LABELS.get(prompt_name, prompt_name)
    if llm_client is False:
        logger.info("LLM agent %s (%s) skipped because LLM analysis is disabled", agent_label, prompt_name)
        return ""
    try:
        client = llm_client or LLMClient()
        prompt = load_prompt(PROMPT_DIR / prompt_name)
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        response = client.chat(prompt, f"请基于以下结构化输入输出分析：\n\n{user_prompt}").strip()
        if response:
            logger.info("LLM agent %s (%s) loaded successfully", agent_label, prompt_name)
        else:
            logger.warning("LLM agent %s (%s) returned an empty response", agent_label, prompt_name)
        return response
    except Exception as exc:
        logger.warning("LLM agent %s (%s) failed: %s", agent_label, prompt_name, exc, exc_info=True)
        return ""


def _llm_max_parallel_requests(default: int = 2) -> int:
    try:
        return max(1, int(os.getenv("LLM_MAX_PARALLEL_REQUESTS", str(default))))
    except ValueError:
        return default


def _compact_news_items(items: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "date": item.get("date", ""),
            "sentiment": classify_news_item_sentiment(item),
            "fetched_via": item.get("fetched_via", ""),
            "related_symbols": item.get("related_symbols", []),
        }
        for item in items[:limit]
    ]


def _compact_financial_reports(items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "title": item.get("title", ""),
            "date": item.get("date", ""),
            "source": item.get("source", ""),
            "financial_signals": item.get("financial_signals", [])[:6],
            "content_excerpt": str(item.get("content_excerpt", ""))[:800],
        }
        for item in items[:limit]
    ]


def _compact_market_risk(market_df: pd.DataFrame, quote_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if market_df.empty or "symbol" not in market_df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for symbol, group in market_df.groupby("symbol"):
        indicators = calculate_indicators(group, quote_metrics.get(str(symbol)))
        rows.append(
            {
                "symbol": symbol,
                "period_return_pct": indicators["period_return_pct"],
                "max_drawdown_pct": indicators["max_drawdown_pct"],
                "amplitude_pct": indicators["amplitude_pct"],
                "ma20": indicators["ma20"],
                "ma60": indicators["ma60"],
                "macd_dif": indicators["macd_dif"],
                "macd_dea": indicators["macd_dea"],
                "volume_ratio_5": indicators["volume_ratio_5"],
                "pe_dynamic": indicators["pe_dynamic"],
                "pb": indicators["pb"],
            }
        )
    return rows


def clean_llm_agent_reply(content: str, agent_name: str) -> str:
    redundant_titles = {
        agent_name,
        f"LLM{agent_name.replace('智能体', '研判')}",
        "LLM新闻研判",
        "LLM技术研判",
        "LLM风险研判",
    }
    cleaned: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        normalized = stripped.strip("#* 　：:")
        if normalized in redundant_titles:
            continue
        cleaned.append(raw_line)
    return "\n".join(cleaned).strip()


def pct(value: float) -> str:
    return f"{value:.2f}%"


def money(value: float) -> str:
    return f"{value:,.2f}"


def format_optional_number(value: float) -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"{value:.2f}"


def format_optional_percent(value: float) -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"{value:.2f}%"


def sentiment_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        label = classify_news_item_sentiment(item)
        counts[label] = counts.get(label, 0) + 1
    return counts


def match_position_news(portfolio: dict[str, Any], news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    position_symbols = {item["symbol"] for item in portfolio.get("positions", [])}
    return [
        item
        for item in news_items
        if position_symbols.intersection(set(item.get("related_symbols", [])))
    ]


def extract_numeric_evidence(news_items: list[dict[str, Any]]) -> list[str]:
    patterns = [
        r"营收[^，。；\n]{0,20}?[-+]?\d+(?:\.\d+)?%",
        r"净利[^，。；\n]{0,20}?[-+]?\d+(?:\.\d+)?%",
        r"净利润[^，。；\n]{0,20}?[-+]?\d+(?:\.\d+)?%",
        r"分红[^，。；\n]{0,20}?\d+(?:\.\d+)?亿元",
        r"回购[^，。；\n]{0,20}?\d+(?:\.\d+)?亿",
        r"净卖出[^，。；\n]{0,20}?\d+(?:\.\d+)?亿元",
        r"提价[^，。；\n]{0,20}?\d+(?:\.\d+)?元",
    ]
    evidence: list[str] = []
    for item in news_items:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        for pattern in patterns:
            for match in re.findall(pattern, text):
                if match not in evidence:
                    evidence.append(match)
        if len(evidence) >= 12:
            break
    return evidence


def compute_technical_metrics(group: pd.DataFrame) -> dict[str, float | str]:
    ordered = group.sort_values("date").copy()
    close = ordered["close"].astype(float)
    high = ordered["high"].astype(float)
    low = ordered["low"].astype(float)
    volume = ordered["volume"].astype(float)
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    period_return = (last_close / first_close - 1) * 100 if first_close else 0.0
    range_pct = (float(high.max()) / float(low.min()) - 1) * 100 if float(low.min()) else 0.0
    rolling_peak = close.cummax()
    drawdown = (close / rolling_peak - 1) * 100
    max_drawdown = float(drawdown.min())
    ma5 = float(close.tail(5).mean())
    ma10 = float(close.tail(10).mean())
    volume_ratio = float(volume.iloc[-1] / volume.tail(min(5, len(volume))).mean()) if len(volume) else 1.0
    close_position = (last_close - float(low.min())) / (float(high.max()) - float(low.min())) * 100
    return {
        "start_date": str(ordered.iloc[0]["date"]),
        "end_date": str(ordered.iloc[-1]["date"]),
        "first_close": first_close,
        "last_close": last_close,
        "period_return": period_return,
        "range_pct": range_pct,
        "max_drawdown": max_drawdown,
        "ma5": ma5,
        "ma10": ma10,
        "volume_ratio": volume_ratio,
        "close_position": close_position,
        "high": float(high.max()),
        "low": float(low.min()),
    }


def run_news_agent(
    portfolio: dict[str, Any],
    news_items: list[dict[str, Any]],
    llm_client: Any | None = None,
) -> AgentResult:
    related_news = match_position_news(portfolio, news_items)
    lines = ["## 新闻解读智能体", ""]

    if not related_news:
        lines.append("未发现与当前持仓直接相关的新闻，报告仅保留市场层面新闻作为背景。")
        return AgentResult(name="新闻解读智能体", content="\n".join(lines))

    counts = sentiment_counts(related_news)
    risk_items = [item for item in related_news if classify_news_item_sentiment(item) == "risk"]
    positive_items = [item for item in related_news if classify_news_item_sentiment(item) == "positive"]
    evidence = extract_numeric_evidence(related_news)

    lines.append(f"样本覆盖：{len(related_news)} 条直接相关资讯；标签分布为 {counts}。")
    lines.append("")
    lines.extend(build_news_sample_table(len(related_news), counts, len(evidence)))
    llm_analysis = _try_llm_agent(
        "news_agent.txt",
        {
            "portfolio": portfolio,
            "news_items": _compact_news_items(related_news),
            "sentiment_counts": counts,
            "numeric_evidence": evidence[:10],
        },
        llm_client,
    )
    if llm_analysis:
        lines.extend(["", "### 新闻研判", clean_llm_agent_reply(llm_analysis, "新闻解读智能体")])
    else:
        lines.append("")
        lines.append("核心主题归纳：")
        if risk_items:
            lines.append(
                f"- 压力主题：{len(risk_items)} 条资讯被识别为风险类，集中在业绩波动、资金净卖出、管理层/治理事件、行业需求与价格体系变化。"
            )
        if positive_items:
            lines.append(
                f"- 支撑主题：{len(positive_items)} 条资讯被识别为正面类，主要涉及分红、回购、扩产/投资、价格变化、季度增长或订单改善。"
            )
        if evidence:
            lines.append("- 可量化线索：" + "；".join(evidence[:8]) + "。")

    lines.append("")
    lines.append("代表性证据链：")
    evidence_items = select_representative_news_evidence(related_news)
    for item in evidence_items:
        title = item.get("title", "未命名新闻")
        label = normalized_sentiment_label(item.get("sentiment_hint"))
        source = item.get("source", "未知来源")
        lines.append(f"- [{label}] {title}（{source}）：{item.get('summary', '')}")
    if len(related_news) > len(evidence_items):
        lines.append(f"- 其余 {len(related_news) - len(evidence_items)} 条直接相关资讯见新闻明细。")

    return AgentResult(name="新闻解读智能体", content="\n".join(lines))


def build_news_sample_table(total: int, counts: dict[str, int], evidence_count: int) -> list[str]:
    return [
        "### 新闻样本概览",
        "",
        "| 样本数 | positive | risk | neutral | 可量化线索 |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {total} | {counts.get('positive', 0)} | {counts.get('risk', 0)} | {counts.get('neutral', 0)} | {evidence_count} |",
    ]


def run_technical_agent(
    market_df: pd.DataFrame,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    llm_client: Any | None = None,
) -> AgentResult:
    quote_metrics = quote_metrics or {}
    lines = ["## 技术观察智能体", ""]

    for symbol, group in market_df.sort_values("date").groupby("symbol"):
        indicators = calculate_indicators(group, quote_metrics.get(symbol))
        classes = classify_indicators(indicators)
        stock_name = str((quote_metrics.get(symbol) or {}).get("name") or "").strip()
        heading = f"{stock_name}（{symbol}）" if stock_name and stock_name != symbol else str(symbol)

        lines.append(f"### {heading}")
        lines.append(
            f"样本区间：{indicators['start_date']} 至 {indicators['end_date']}。本模块计算趋势、动量、波动、量能、估值五类指标，共 {len(format_indicator_table(indicators))} 项。"
        )
        llm_analysis = _try_llm_agent(
            "technical_agent.txt",
            {"symbol": symbol, "indicators": indicators, "classes": classes},
            llm_client,
        )
        if not llm_analysis:
            lines.extend(build_technical_status_lines(group, indicators, classes))
        lines.append("")
        lines.append("| 指标 | 数值 | 含义 |")
        lines.append("| --- | ---: | --- |")
        lines.extend(format_indicator_table(indicators))
        if llm_analysis:
            lines.extend(["", "### 技术研判", clean_llm_agent_reply(llm_analysis, "技术观察智能体")])
        else:
            lines.extend(build_rule_technical_analysis(group, indicators, classes))
        lines.append("")

    return AgentResult(name="技术观察智能体", content="\n".join(lines).rstrip())


def build_rule_technical_analysis(
    group: pd.DataFrame,
    indicators: dict[str, Any],
    classes: dict[str, str],
) -> list[str]:
    lines = ["", "综合解读："]
    lines.append(
        f"- 趋势：收盘 {indicators['close']:.2f}，MA5/MA20/MA60 分别为 {indicators['ma5']:.2f}/{indicators['ma20']:.2f}/{indicators['ma60']:.2f}，结构判断为{classes['trend']}。"
    )
    lines.append(
        f"- 动量：MACD DIF {indicators['macd_dif']:.2f}、DEA {indicators['macd_dea']:.2f}、柱值 {indicators['macd_bar']:.2f}，判断为{classes['macd']}；RSI6/12/24 为 {indicators['rsi6']:.2f}/{indicators['rsi12']:.2f}/{indicators['rsi24']:.2f}，对应{classes['rsi']}、{classes['medium_rsi']}。"
    )
    lines.append(
        f"- 摆动：KDJ K/D/J 为 {indicators['kdj_k']:.2f}/{indicators['kdj_d']:.2f}/{indicators['kdj_j']:.2f}，处于{classes['kdj']}；BOLL %B 为 {indicators['boll_percent_b']:.2f}，价格{classes['boll']}。"
    )
    lines.append(
        f"- 波动：ATR14 为 {indicators['atr14']:.2f}，20日年化波动率 {indicators['volatility20_pct']:.2f}%，区间最大回撤 {indicators['max_drawdown_pct']:.2f}%，说明短期波动压力需要和新闻/公告同步观察。"
    )
    lines.append(
        f"- 量能：成交量/5日均量为 {indicators['volume_ratio_5']:.2f}，换手率 {format_optional_percent(indicators['turnover_rate_pct'])}，属于{classes['volume']}。若价格方向与量能背离，信号可信度下降。"
    )
    lines.append(
        f"- 估值：动态PE {format_optional_number(indicators['pe_dynamic'])}，PB {format_optional_number(indicators['pb'])}，判断为{classes['valuation']}；估值指标需要和利润增速、现金分红、行业景气度、公司竞争壁垒一起解释。"
    )
    lines.extend(analyze_chart_trends(group, indicators))
    lines.append(
        "- 技术面结论：若趋势、MACD、RSI、量能同时改善，说明短线修复更有一致性；若估值仍高而业绩新闻偏弱，则技术反弹更容易受基本面预期压制。"
    )
    return lines


def build_technical_status_lines(
    group: pd.DataFrame,
    indicators: dict[str, Any],
    classes: dict[str, str],
) -> list[str]:
    df = group.sort_values("date").copy()
    close = df["close"].astype(float).reset_index(drop=True)
    volume = df["volume"].astype(float).reset_index(drop=True)
    recent = min(10, len(close))
    recent_change = (float(close.iloc[-1]) / float(close.iloc[-recent]) - 1) * 100 if recent > 1 and close.iloc[-recent] else 0.0
    volume_ratio = indicators["volume_ratio_5"]

    if recent_change <= -5 and indicators["close_position_pct"] >= 65:
        status = "高位回调"
    elif indicators["close"] < indicators["ma20"] and indicators["macd_dif"] < indicators["macd_dea"]:
        status = "技术转弱"
    elif indicators["close"] > indicators["ma5"] > indicators["ma20"] and indicators["macd_dif"] > indicators["macd_dea"]:
        status = "趋势偏强"
    else:
        status = "趋势分化"

    volume_view = "放量" if volume_ratio >= 1.3 else "缩量" if volume_ratio <= 0.75 else "量能平稳"
    latest_volume = float(volume.iloc[-1]) if len(volume) else 0.0
    return [
        "",
        f"技术状态：{status}",
        "关键证据：",
        f"- 近 {recent} 个交易日收盘价变化 {recent_change:.2f}%，价格区间分位 {indicators['close_position_pct']:.2f}%。",
        f"- MACD DIF/DEA 为 {indicators['macd_dif']:.2f}/{indicators['macd_dea']:.2f}，动能判断为{classes['macd']}。",
        f"- 成交量/5日均量为 {volume_ratio:.2f}，当前量能为{volume_view}；最新成交量 {latest_volume:,.0f}。",
        "关键观察位：",
        f"- 上方确认位：MA20 {indicators['ma20']:.2f}，若放量站回，修复信号更有说服力。",
        f"- 下方风险位：MA60 {indicators['ma60']:.2f} / BOLL下轨 {indicators['boll_lower']:.2f}。",
    ]


def trend_word(value: float, threshold: float = 0.0) -> str:
    if value > threshold:
        return "上行"
    if value < -threshold:
        return "下行"
    return "走平"


def analyze_chart_trends(group: pd.DataFrame, indicators: dict[str, Any]) -> list[str]:
    df = group.sort_values("date").copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    amount = df.get("amount", ((open_ + close) / 2) * volume).astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = (dif - dea) * 2

    recent = min(10, len(df))
    prev = min(10, max(len(df) - recent, 0))
    recent_close_change = (close.iloc[-1] / close.iloc[-recent] - 1) * 100 if recent > 1 and close.iloc[-recent] else 0.0
    recent_macd_delta = float(macd_bar.iloc[-1] - macd_bar.iloc[-recent]) if recent > 1 else 0.0
    recent_amount_avg = float(amount.tail(recent).mean()) if recent else 0.0
    prev_amount_avg = float(amount.iloc[-recent - prev : -recent].mean()) if prev else recent_amount_avg
    amount_delta = (recent_amount_avg / prev_amount_avg - 1) * 100 if prev_amount_avg else 0.0
    red_days = int((close.tail(recent) >= open_.tail(recent)).sum())
    green_days = recent - red_days
    close_now = float(close.iloc[-1])
    ma5 = float(close.rolling(5, min_periods=1).mean().iloc[-1])
    ma20 = float(close.rolling(20, min_periods=1).mean().iloc[-1])
    ma60 = float(close.rolling(60, min_periods=1).mean().iloc[-1])
    ma5_prev = float(close.rolling(5, min_periods=1).mean().iloc[-recent])
    ma20_prev = float(close.rolling(20, min_periods=1).mean().iloc[-recent])
    dif_now = float(dif.iloc[-1])
    dea_now = float(dea.iloc[-1])
    dif_prev = float(dif.iloc[-recent])
    dea_prev = float(dea.iloc[-recent])
    hist_now = float(macd_bar.iloc[-1])
    hist_prev = float(macd_bar.iloc[-recent])
    latest_volume_ratio = float(volume.iloc[-1] / volume.tail(min(5, len(volume))).mean())
    price_above_ma = close_now > ma5 > ma20
    price_below_ma = close_now < ma5 < ma20
    macd_gold = dif_prev <= dea_prev and dif_now > dea_now
    macd_dead = dif_prev >= dea_prev and dif_now < dea_now
    hist_expanding_up = hist_now > 0 and hist_now > hist_prev
    hist_shrinking_down = hist_now < 0 and hist_now > hist_prev

    if price_above_ma and ma5 >= ma5_prev and ma20 >= ma20_prev:
        ma_view = "短期均线在中期均线上方且均线抬升，按最简单的 MA 原理，短线趋势偏修复。"
    elif price_below_ma and ma5 <= ma5_prev and ma20 <= ma20_prev:
        ma_view = "价格在短中期均线下方且均线下压，按最简单的 MA 原理，短线趋势仍偏弱。"
    else:
        ma_view = "价格和均线关系分化，说明趋势未形成单边一致性，更像震荡或弱修复。"

    if macd_gold:
        macd_view = "DIF 上穿 DEA，出现 MACD 金叉，动能由弱转强的概率上升。"
    elif macd_dead:
        macd_view = "DIF 下穿 DEA，出现 MACD 死叉，短期动能转弱。"
    elif hist_expanding_up:
        macd_view = "DIF 位于 DEA 上方且红柱扩张，短线动能延续改善。"
    elif hist_shrinking_down:
        macd_view = "MACD 仍在弱区但绿柱收窄，说明下行动能有减弱迹象，还不能等同于趋势反转。"
    elif hist_now < hist_prev:
        macd_view = "MACD 柱值走弱，说明近期动能边际变差。"
    else:
        macd_view = "MACD 变化不强，更多体现为弱震荡，需要等待 DIF/DEA 和柱值给出更明确方向。"

    if amount_delta >= 10 and recent_close_change > 0:
        volume_view = "成交额放大且价格上行，说明上涨有量能配合，修复信号可信度提高。"
    elif amount_delta >= 10 and recent_close_change <= 0:
        volume_view = "成交额放大但价格没有同步上行，说明分歧或抛压增加，需要警惕放量滞涨。"
    elif amount_delta <= -10 and recent_close_change > 0:
        volume_view = "价格上行但成交额萎缩，说明反弹承接力度有限，持续性需要继续验证。"
    elif amount_delta <= -10 and recent_close_change <= 0:
        volume_view = "价格走弱且成交额萎缩，说明资金关注度下降，也可能意味着抛压暂时缓和。"
    else:
        volume_view = "成交额变化不极端，量能对方向的确认力度一般。"

    if "偏修复" in ma_view and ("改善" in macd_view or "金叉" in macd_view) and amount_delta > 0:
        forward_view = "未来短线更偏向延续修复观察，但前提是价格不再跌回 MA20 下方。"
    elif "偏弱" in ma_view and ("转弱" in macd_view or "走弱" in macd_view or "死叉" in macd_view):
        forward_view = "未来短线仍需按弱势震荡看待，优先观察能否止跌并重新站回 MA5/MA20。"
    else:
        forward_view = "未来走势暂不宜给单边判断，更适合观察 MA20、DIF/DEA 交叉和成交量是否共同确认。"

    lines = ["- 图表读取："]
    lines.append(
        f"  - K线/均线：近 {recent} 个交易日收盘价变化 {recent_close_change:.2f}%，MA5/MA20/MA60 位置为 {indicators['ma5']:.2f}/{indicators['ma20']:.2f}/{indicators['ma60']:.2f}，短线价格相对均线呈{trend_word(recent_close_change, 0.5)}。"
    )
    lines.append(f"  - MA 初步判断：{ma_view}")
    lines.append(
        f"  - MACD：柱值从近 {recent} 日起点到最新变化 {recent_macd_delta:.2f}，DIF {indicators['macd_dif']:.2f}、DEA {indicators['macd_dea']:.2f}，动能呈{trend_word(recent_macd_delta, 0.02)}。"
    )
    lines.append(f"  - MACD 初步判断：{macd_view}")
    lines.append(
        f"  - 成交量/成交额：近 {recent} 日平均成交额较前段变化 {amount_delta:.2f}%，最新成交量/5日均量 {latest_volume_ratio:.2f}，红K {red_days} 天、绿K {green_days} 天，量价配合呈{trend_word(amount_delta, 5)}。"
    )
    lines.append(f"  - 成交量含义：{volume_view}")
    lines.append(f"  - 未来走势初步判断：{forward_view}")
    return lines


def run_risk_agent(
    portfolio: dict[str, Any],
    news_items: list[dict[str, Any]],
    market_df: pd.DataFrame,
    financial_reports: list[dict[str, Any]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    llm_client: Any | None = None,
) -> AgentResult:
    positions = portfolio.get("positions", [])
    total_value = sum(float(item["quantity"]) * float(item["market_price"]) for item in positions)
    related_news = match_position_news(portfolio, news_items)
    risk_news = [item for item in related_news if classify_news_item_sentiment(item) == "risk"]
    positive_news = [item for item in related_news if classify_news_item_sentiment(item) == "positive"]
    evidence = extract_numeric_evidence(related_news)

    lines = ["## 风险提示智能体", ""]
    lines.append("风险分层：")
    lines.extend(build_risk_basis_table(positions, total_value, related_news, risk_news, positive_news))

    llm_analysis = _try_llm_agent(
        "risk_agent.txt",
        {
            "portfolio": portfolio,
            "risk_news": _compact_news_items(risk_news[:10]),
            "positive_news": _compact_news_items(positive_news[:10]),
            "financial_reports": _compact_financial_reports(financial_reports or []),
            "market_risk": _compact_market_risk(market_df, quote_metrics or {}),
        },
        llm_client,
    )
    if llm_analysis:
        lines.extend(["### 风险研判", clean_llm_agent_reply(llm_analysis, "风险提示智能体")])
        return AgentResult(name="风险提示智能体", content="\n".join(lines))

    for item in positions:
        value = float(item["quantity"]) * float(item["market_price"])
        weight = value / total_value * 100 if total_value else 0.0
        pnl_pct = (float(item["market_price"]) / float(item["cost_price"]) - 1) * 100
        lines.append(
            f"- 持仓暴露：{item['name']}（{item['symbol']}）市值 {money(value)}，组合权重 {pct(weight)}，相对成本收益 {pct(pnl_pct)}。"
        )
        if weight >= 70:
            lines.append("  推理：单一标的权重较高，新闻、财报或价格波动会直接主导组合净值变化。")
        if pnl_pct < 0:
            lines.append("  推理：当前价格低于样例成本，若基本面负面线索继续增加，心理止损压力和波动承受能力需要提前评估。")

    for symbol, group in market_df.groupby("symbol"):
        indicators = calculate_indicators(group, (quote_metrics or {}).get(symbol))
        lines.append(
            f"- 市场波动：{symbol} 样本区间收益 {pct(indicators['period_return_pct'])}，最大回撤 {pct(indicators['max_drawdown_pct'])}，振幅 {pct(indicators['amplitude_pct'])}，动态PE {format_optional_number(indicators['pe_dynamic'])}，PB {format_optional_number(indicators['pb'])}。"
        )
        if indicators["max_drawdown_pct"] <= -3 or indicators["amplitude_pct"] >= 8:
            lines.append("  推理：价格波动已经具备短期风险暴露特征，需要和新闻事件时间点交叉验证。")
        else:
            lines.append("  推理：样本期价格波动不极端，但估值指标和基本面预期仍可能放大后续价格反应。")

    lines.append(
        f"- 信息面压力：直接相关新闻 {len(related_news)} 条，其中风险类 {len(risk_news)} 条、正面类 {len(positive_news)} 条。"
    )
    if risk_news:
        lines.append("  主要风险标题：")
        for item in risk_news[:6]:
            lines.append(f"  - {item.get('title', '')}")
    if evidence:
        lines.append("  可量化风险/支撑线索：" + "；".join(evidence[:10]) + "。")

    if financial_reports:
        report_titles = "；".join(item.get("title", "") for item in financial_reports[:5])
        lines.append(f"- 财报公告压力测试：最近财务公告包含 {report_titles}。")
        lines.append(
            "  推理：年度报告、季度报告、利润分配和业绩说明会公告同时出现时，应重点核对收入增速、净利润增速、现金分红、回购安排和管理层回应。"
        )

    lines.append("")
    lines.append("需要继续追踪的触发器：")
    lines.append("- 后续公告是否修正经营目标、分红安排、回购进度或管理层职责。")
    lines.append("- 行业需求、产品价格、订单/库存变化是否和财报中的收入/利润变化相互印证。")
    lines.append("- 资金流向新闻是否从单日净卖出演变为连续多日同向变化。")

    return AgentResult(name="风险提示智能体", content="\n".join(lines))


def build_risk_basis_table(
    positions: list[dict[str, Any]],
    total_value: float,
    related_news: list[dict[str, Any]],
    risk_news: list[dict[str, Any]],
    positive_news: list[dict[str, Any]],
) -> list[str]:
    max_weight = 0.0
    if total_value:
        max_weight = max(
            (
                float(item.get("quantity", 0)) * float(item.get("market_price", 0)) / total_value * 100
                for item in positions
            ),
            default=0.0,
        )
    return [
        "",
        "### 风险基础数据",
        "",
        "| 持仓数量 | 最大单票权重 | 相关新闻 | risk新闻 | positive新闻 |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {len(positions)} | {max_weight:.2f}% | {len(related_news)} | {len(risk_news)} | {len(positive_news)} |",
        "",
    ]


def run_financial_report_agent(financial_reports: list[dict[str, Any]], news_items: list[dict[str, Any]] | None = None) -> AgentResult:
    lines = ["## 财报公告智能体", ""]
    if not financial_reports:
        lines.append("未抓取到最近财报或财务相关公告，需要检查公告源或扩大抓取周期。")
        return AgentResult(name="财报公告智能体", content="\n".join(lines))

    annual = [item for item in financial_reports if "年度报告" in item.get("title", "")]
    quarterly = [item for item in financial_reports if "季度报告" in item.get("title", "")]
    distribution = [item for item in financial_reports if "分配" in item.get("title", "") or "分红" in item.get("title", "")]
    meeting = [item for item in financial_reports if "业绩说明" in item.get("title", "")]
    evidence = extract_numeric_evidence(news_items or [])

    lines.append(f"公告样本：{len(financial_reports)} 条，其中年度报告 {len(annual)} 条、季度报告 {len(quarterly)} 条、分红/利润分配 {len(distribution)} 条、业绩说明 {len(meeting)} 条。")
    lines.append("")
    lines.append("公告解读框架：")
    if annual:
        lines.append(f"- 年报线索：最新年报相关公告为 {annual[0].get('date', '')}《{annual[0].get('title', '')}》。")
    if quarterly:
        lines.append(f"- 季报线索：最新季报相关公告为 {quarterly[0].get('date', '')}《{quarterly[0].get('title', '')}》。")
    if distribution:
        lines.append(f"- 股东回报线索：存在利润分配/分红安排公告，说明现金回报是当前市场关注点之一。")
    if meeting:
        lines.append(f"- 沟通线索：业绩说明会公告意味着管理层将回应经营质量、增长目标、渠道库存和价格体系等问题。")
    if evidence:
        lines.append("- 新闻中提取到的财务数字：" + "；".join(evidence[:10]) + "。")

    content_reports = [item for item in financial_reports if item.get("content_excerpt")]
    signal_reports = [item for item in financial_reports if item.get("financial_signals")]
    lines.append("")
    lines.append("公告正文读取结果：")
    if content_reports:
        lines.append(
            f"- 已读取 {len(content_reports)} 条公告正文片段；其中 {len(signal_reports)} 条提取到收入、利润、现金流、分红或主营业务等文本线索。"
        )
        for item in content_reports[:4]:
            title = item.get("title", "未命名公告")
            pages = item.get("content_pages_read", 0)
            signals = item.get("financial_signals", [])
            lines.append(f"- 《{title}》：已读取 {pages} 页公告正文。")
            if signals:
                for signal in signals[:4]:
                    lines.append(f"  - 正文线索：{signal}")
            else:
                excerpt = item.get("content_excerpt", "")[:160]
                if excerpt:
                    lines.append(f"  - 正文摘要：{excerpt}...")
    else:
        lines.append("- 未能读取公告正文，只能基于公告标题和新闻侧财务线索做弱分析。")

    lines.append("")
    lines.append("分析师式追问：")
    lines.append("- 年度收入和净利润变化是否来自销量、价格、产品结构、渠道投放或费用率变化。")
    lines.append("- 高分红和回购是否足以抵消增长放缓带来的估值压力。")
    lines.append("- 季报改善是否具有持续性，还是基数、发货节奏或费用确认带来的阶段性变化。")

    return AgentResult(name="财报公告智能体", content="\n".join(lines))


def run_all_agents(
    portfolio: dict[str, Any],
    news_items: list[dict[str, Any]],
    market_df: pd.DataFrame,
    financial_reports: list[dict[str, Any]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    llm_client: Any | None = None,
) -> list[AgentResult]:
    if llm_client is False:
        results = [
            run_news_agent(portfolio, news_items, llm_client=False),
            run_technical_agent(market_df, quote_metrics=quote_metrics, llm_client=False),
            run_risk_agent(
                portfolio,
                news_items,
                market_df,
                financial_reports=financial_reports,
                quote_metrics=quote_metrics,
                llm_client=False,
            ),
        ]
    else:
        with ThreadPoolExecutor(max_workers=_llm_max_parallel_requests()) as executor:
            news_future = executor.submit(run_news_agent, portfolio, news_items, llm_client)
            technical_future = executor.submit(run_technical_agent, market_df, quote_metrics, llm_client)
            risk_future = executor.submit(
                run_risk_agent,
                portfolio,
                news_items,
                market_df,
                financial_reports,
                quote_metrics,
                llm_client,
            )
            results = [news_future.result(), technical_future.result(), risk_future.result()]
    if financial_reports is not None:
        results.insert(1, run_financial_report_agent(financial_reports, news_items=news_items))
    return results
