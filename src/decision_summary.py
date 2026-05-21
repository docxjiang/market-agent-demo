from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm_client import LLMClient
from src.news_evidence import classify_news_item_sentiment
from src.technical_indicators import calculate_indicators


DECISION_SUMMARY_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "decision_summary_agent.txt"
logger = logging.getLogger(__name__)


def load_decision_summary_prompt() -> str:
    return DECISION_SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")


def build_decision_summary(
    portfolio: dict[str, Any],
    market_df: pd.DataFrame,
    news_items: list[dict[str, Any]],
    financial_reports: list[dict[str, Any]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    prefer_llm: bool = True,
) -> dict[str, Any]:
    financial_reports = financial_reports or []
    quote_metrics = quote_metrics or {}
    if prefer_llm:
        try:
            client = llm_client or LLMClient()
            summary = _build_decision_summary_with_llm(
                client,
                portfolio,
                market_df,
                news_items,
                financial_reports,
                quote_metrics,
            )
            summary["generated_by"] = "llm"
            logger.info("Decision summary LLM (AI信息整合) loaded successfully")
            return summary
        except Exception as exc:
            logger.warning("Decision summary LLM (AI信息整合) failed: %s", exc, exc_info=True)
    else:
        logger.info("Decision summary LLM (AI信息整合) skipped because LLM analysis is disabled")

    summary = build_rule_decision_summary(
        portfolio,
        market_df,
        news_items,
        financial_reports=financial_reports,
        quote_metrics=quote_metrics,
    )
    summary["generated_by"] = "rules"
    return summary


def build_rule_decision_summary(
    portfolio: dict[str, Any],
    market_df: pd.DataFrame,
    news_items: list[dict[str, Any]],
    financial_reports: list[dict[str, Any]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    financial_reports = financial_reports or []
    quote_metrics = quote_metrics or {}
    usable_news = [item for item in news_items if not _is_fallback_news(item)]
    technical = _technical_view(market_df, quote_metrics)
    positive_factors = _positive_factors(usable_news, financial_reports)
    negative_factors = _negative_factors(usable_news, technical)
    holder_focus = _holder_focus(portfolio, technical)
    state = _state_label(positive_factors, negative_factors, technical)

    return {
        "state": state,
        "one_line": _one_line(state, positive_factors, negative_factors, technical),
        "top_judgements": _top_judgements(positive_factors, negative_factors, technical, holder_focus),
        "positive_factors": positive_factors[:3] or ["暂未识别到足以进入主判断的明确利多信息。"],
        "negative_factors": negative_factors[:3] or ["暂未识别到足以进入主判断的明确利空信息。"],
        "holder_focus": holder_focus[:3],
        "buyer_focus": _buyer_focus(technical),
        "seller_focus": _seller_focus(technical),
        "bullish_triggers": _bullish_triggers(technical),
        "bearish_triggers": _bearish_triggers(technical),
        "generated_by": "rules",
    }


def _build_decision_summary_with_llm(
    client: Any,
    portfolio: dict[str, Any],
    market_df: pd.DataFrame,
    news_items: list[dict[str, Any]],
    financial_reports: list[dict[str, Any]],
    quote_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    system_prompt = load_decision_summary_prompt()
    user_prompt = json.dumps(
        {
            "portfolio": portfolio,
            "market": _market_payload(market_df, quote_metrics),
            "news": _news_payload(news_items),
            "financial_reports": _financial_payload(financial_reports),
        },
        ensure_ascii=False,
        indent=2,
    )
    raw = client.chat(system_prompt, f"请基于以下输入生成 AI决策摘要 JSON：\n\n{user_prompt}")
    return _coerce_summary(_parse_json_object(raw))


def decision_summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "## AI决策摘要",
        "",
        "### 当前状态",
        f"- **{summary['state']}**：{summary['one_line']}",
        "",
        "### 最重要的3个判断",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(summary["top_judgements"][:3], start=1))
    lines.extend(["", "### 关键利多"])
    lines.extend(f"- {item}" for item in summary["positive_factors"][:3])
    lines.extend(["", "### 关键利空"])
    lines.extend(f"- {item}" for item in summary["negative_factors"][:3])
    lines.extend(["", "### 对不同用户的关注点", "- 已持有："])
    lines.extend(f"  - {item}" for item in summary["holder_focus"][:3])
    lines.append("- 准备买入：")
    lines.extend(f"  - {item}" for item in summary["buyer_focus"][:3])
    lines.append("- 准备卖出：")
    lines.extend(f"  - {item}" for item in summary["seller_focus"][:3])
    lines.extend(["", "### 后续触发条件", "- 偏积极："])
    lines.extend(f"  - {item}" for item in summary["bullish_triggers"][:3])
    lines.append("- 偏消极：")
    lines.extend(f"  - {item}" for item in summary["bearish_triggers"][:3])
    return "\n".join(lines)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("decision summary response must be a JSON object")
    return parsed


def _coerce_summary(summary: dict[str, Any]) -> dict[str, Any]:
    required_list_fields = [
        "top_judgements",
        "positive_factors",
        "negative_factors",
        "holder_focus",
        "buyer_focus",
        "seller_focus",
        "bullish_triggers",
        "bearish_triggers",
    ]
    result = {
        "state": str(summary.get("state") or "信息中性"),
        "one_line": str(summary.get("one_line") or "当前有效信号不够集中，需要继续观察。"),
    }
    for field in required_list_fields:
        value = summary.get(field)
        if isinstance(value, list):
            result[field] = [str(item) for item in value if str(item).strip()][:4]
        elif value:
            result[field] = [str(value)]
        else:
            result[field] = []
    if not result["top_judgements"]:
        raise ValueError("decision summary response missing top_judgements")
    return result


def _market_payload(market_df: pd.DataFrame, quote_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if market_df.empty or "symbol" not in market_df.columns:
        return {"available": False}
    payload: dict[str, Any] = {"available": True, "symbols": []}
    for symbol, group in market_df.sort_values("date").groupby("symbol"):
        indicators = calculate_indicators(group, quote_metrics.get(str(symbol)))
        payload["symbols"].append(
            {
                "symbol": str(symbol),
                "start_date": indicators["start_date"],
                "end_date": indicators["end_date"],
                "close": indicators["close"],
                "period_return_pct": indicators["period_return_pct"],
                "max_drawdown_pct": indicators["max_drawdown_pct"],
                "close_position_pct": indicators["close_position_pct"],
                "ma20": indicators["ma20"],
                "ma60": indicators["ma60"],
                "macd_dif": indicators["macd_dif"],
                "macd_dea": indicators["macd_dea"],
                "macd_bar": indicators["macd_bar"],
                "volume_ratio_5": indicators["volume_ratio_5"],
                "boll_lower": indicators["boll_lower"],
            }
        )
    return payload


def _news_payload(news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for item in news_items[:40]:
        payload.append(
            {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "date": item.get("date", ""),
                "sentiment": classify_news_item_sentiment(item),
                "fetched_via": item.get("fetched_via", ""),
                "related_symbols": item.get("related_symbols", []),
            }
        )
    return payload


def _financial_payload(financial_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": item.get("title", ""),
            "date": item.get("date", ""),
            "source": item.get("source", ""),
            "financial_signals": item.get("financial_signals", [])[:6],
            "content_excerpt": str(item.get("content_excerpt", ""))[:800],
        }
        for item in financial_reports[:8]
    ]


def _technical_view(market_df: pd.DataFrame, quote_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if market_df.empty or "symbol" not in market_df.columns:
        return {"status": "趋势不明", "weak": False, "strong": False, "evidence": []}

    symbol = str(market_df.sort_values("date").iloc[-1]["symbol"])
    group = market_df[market_df["symbol"] == symbol]
    indicators = calculate_indicators(group, quote_metrics.get(symbol))
    ordered = group.sort_values("date")
    close = ordered["close"].astype(float).reset_index(drop=True)
    recent_window = min(10, len(close))
    recent_change = (float(close.iloc[-1]) / float(close.iloc[-recent_window]) - 1) * 100 if recent_window > 1 and close.iloc[-recent_window] else 0.0
    weak = recent_change <= -5 or indicators["macd_dif"] < indicators["macd_dea"] or indicators["close"] < indicators["ma20"]
    strong = indicators["close"] > indicators["ma5"] > indicators["ma20"] and indicators["macd_dif"] > indicators["macd_dea"]

    if weak and indicators["close_position_pct"] >= 65:
        status = "高位回调"
    elif weak:
        status = "技术转弱"
    elif strong:
        status = "趋势偏强"
    else:
        status = "趋势分化"

    evidence = [
        f"近{recent_window}日涨跌幅 {recent_change:.2f}%。",
        f"MACD DIF/DEA 为 {indicators['macd_dif']:.2f}/{indicators['macd_dea']:.2f}。",
        f"MA20/MA60 为 {indicators['ma20']:.2f}/{indicators['ma60']:.2f}。",
    ]
    return {
        "status": status,
        "weak": weak,
        "strong": strong,
        "recent_change_pct": recent_change,
        "ma20": indicators["ma20"],
        "ma60": indicators["ma60"],
        "boll_lower": indicators["boll_lower"],
        "evidence": evidence,
    }


def _positive_factors(news_items: list[dict[str, Any]], financial_reports: list[dict[str, Any]]) -> list[str]:
    factors: list[str] = []
    for item in news_items:
        if classify_news_item_sentiment(item) == "positive":
            factors.append(_news_factor(item))
        if len(factors) >= 3:
            break

    for report in financial_reports:
        title = str(report.get("title", ""))
        signals = "；".join(str(signal) for signal in report.get("financial_signals", [])[:2])
        text = f"{title} {signals}"
        if any(keyword in text for keyword in ["分红", "利润分配", "回购", "净利润", "增长", "预增"]):
            factors.append(f"公告线索：{title}" + (f"（{signals}）" if signals else ""))
        if len(factors) >= 3:
            break
    return _dedupe(factors)


def _negative_factors(news_items: list[dict[str, Any]], technical: dict[str, Any]) -> list[str]:
    factors = [_news_factor(item) for item in news_items if classify_news_item_sentiment(item) == "risk"]
    if technical.get("weak"):
        factors.append(f"技术面：{technical['status']}，{technical['evidence'][0]}")
    return _dedupe(factors)


def _holder_focus(portfolio: dict[str, Any], technical: dict[str, Any]) -> list[str]:
    positions = portfolio.get("positions", [])
    total_value = sum(float(item.get("quantity", 0)) * float(item.get("market_price", 0)) for item in positions)
    focus: list[str] = []
    for item in positions:
        value = float(item.get("quantity", 0)) * float(item.get("market_price", 0))
        weight = value / total_value * 100 if total_value else 0.0
        if weight >= 70:
            focus.append(f"单一标的仓位约 {weight:.0f}%，个股波动会直接主导组合净值。")
    if technical.get("weak"):
        focus.append("重点观察回调是否扩大，而不是只看单条利好新闻。")
    return focus or ["关注持仓标的的新闻、公告和技术状态是否相互印证。"]


def _buyer_focus(technical: dict[str, Any]) -> list[str]:
    if technical.get("weak"):
        return ["避免只因业绩或新闻利好追高，优先等待技术企稳。", "观察是否重新站回 MA20 并伴随成交量改善。"]
    return ["观察利多是否继续被公告或资金面确认。", "避免在信息不足时只依据单一新闻做判断。"]


def _seller_focus(technical: dict[str, Any]) -> list[str]:
    return ["若价格跌破关键均线且资金面继续走弱，需要重新评估持仓风险。", "若后续公告弱化增长预期，应降低对短线修复的依赖。"]


def _bullish_triggers(technical: dict[str, Any]) -> list[str]:
    return [
        f"放量站回 MA20（约 {technical.get('ma20', 0):.2f}）。",
        "后续公告继续验证利润增长、分红或订单改善。",
        "风险新闻减少，资金流出压力缓和。",
    ]


def _bearish_triggers(technical: dict[str, Any]) -> list[str]:
    return [
        f"跌破 MA60（约 {technical.get('ma60', 0):.2f}）或 BOLL 下轨（约 {technical.get('boll_lower', 0):.2f}）。",
        "资金净卖出或负面新闻连续出现。",
        "业绩说明会或公告对增长持续性的表述转弱。",
    ]


def _state_label(positive_factors: list[str], negative_factors: list[str], technical: dict[str, Any]) -> str:
    if positive_factors and (negative_factors or technical.get("weak")):
        return "分歧偏谨慎"
    if positive_factors and technical.get("strong"):
        return "偏积极"
    if negative_factors or technical.get("weak"):
        return "风险升高"
    return "信息中性"


def _one_line(state: str, positive_factors: list[str], negative_factors: list[str], technical: dict[str, Any]) -> str:
    if state == "分歧偏谨慎":
        return f"基本面催化存在，但技术状态为{technical['status']}，需要用触发条件验证后续方向。"
    if state == "偏积极":
        return "新闻与技术信号相对一致，但仍需跟踪公告和成交量确认。"
    if state == "风险升高":
        return "当前负面线索或技术压力更突出，需优先控制回撤和事件风险。"
    return "当前有效信号不够集中，适合继续观察新闻、公告和量价变化。"


def _top_judgements(
    positive_factors: list[str],
    negative_factors: list[str],
    technical: dict[str, Any],
    holder_focus: list[str],
) -> list[str]:
    items = [
        positive_factors[0] if positive_factors else "基本面催化尚未形成强一致证据。",
        f"技术面：{technical['status']}，{technical['evidence'][0]}" if technical.get("evidence") else "技术面数据不足。",
        holder_focus[0],
    ]
    if negative_factors:
        items[1] = negative_factors[0]
    return items


def _news_factor(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "未命名新闻")
    source = str(item.get("source") or item.get("fetched_via") or "未知来源")
    return f"新闻线索：{title}（{source}）"


def _is_fallback_news(item: dict[str, Any]) -> bool:
    return str(item.get("fetched_via") or "").startswith("local_")


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
