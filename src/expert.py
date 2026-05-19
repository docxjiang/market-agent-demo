from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.llm_client import LLMClient, LLMConfig
from src.technical_indicators import calculate_indicators, classify_indicators
from src.utils import ensure_disclaimer


EXPERT_SYSTEM_PROMPT = """你是一个严谨的A股持仓解读专家。

你会读取当前股票的持仓、行情、技术指标、新闻、财报公告正文线索和多智能体报告，然后回答用户问题。

硬性要求：
- 不输出买入、卖出、持有、加仓、减仓等交易建议。
- 不承诺确定性涨跌。
- 明确区分事实、推理和需要继续观察的变量。
- 回答要结合当前报告数据，不要泛泛而谈。
- 输出中文 Markdown。
- 结尾必须保留“不构成投资建议”的提示。
"""


def build_expert_context(result: dict[str, Any]) -> str:
    portfolio = result["portfolio"]
    market_df: pd.DataFrame = result["market_df"]
    news_items = result["news_items"]
    financial_reports = result["financial_reports"]
    agent_results = result["agent_results"]
    quote_metrics = result.get("quote_metrics", {})

    sections = [
        "## 持仓与股票资料",
        json.dumps(portfolio, ensure_ascii=False, indent=2),
        "",
        "## 行情与技术摘要",
        build_market_context(market_df, quote_metrics),
        "",
        "## 近期新闻标题",
        "\n".join(
            f"- [{item.get('sentiment_hint', 'neutral')}] {item.get('date', '')} {item.get('title', '')}"
            for item in news_items[:30]
        ),
        "",
        "## 财报公告正文线索",
        build_financial_context(financial_reports),
        "",
        "## 多智能体报告",
        "\n\n".join(item.content for item in agent_results),
        "",
        "## 完整报告节选",
        result["report"][:12000],
    ]
    return "\n".join(sections)


def build_market_context(market_df: pd.DataFrame, quote_metrics: dict[str, dict[str, Any]]) -> str:
    if market_df.empty:
        return "当前没有可用行情数据。"

    lines: list[str] = []
    for symbol, group in market_df.sort_values("date").groupby("symbol"):
        indicators = calculate_indicators(group, quote_metrics.get(symbol))
        classes = classify_indicators(indicators)
        lines.extend(
            [
                f"### {symbol}",
                f"- 区间：{indicators['start_date']} 至 {indicators['end_date']}",
                f"- 收盘价：{indicators['close']:.2f}；区间收益：{indicators['period_return_pct']:.2f}%；最大回撤：{indicators['max_drawdown_pct']:.2f}%",
                f"- MA5/MA20/MA60：{indicators['ma5']:.2f}/{indicators['ma20']:.2f}/{indicators['ma60']:.2f}；趋势判断：{classes['trend']}",
                f"- MACD DIF/DEA/柱：{indicators['macd_dif']:.2f}/{indicators['macd_dea']:.2f}/{indicators['macd_bar']:.2f}；动量判断：{classes['macd']}",
                f"- 成交量/5日均量：{indicators['volume_ratio_5']:.2f}；量能判断：{classes['volume']}",
                f"- RSI6/12/24：{indicators['rsi6']:.2f}/{indicators['rsi12']:.2f}/{indicators['rsi24']:.2f}",
            ]
        )
    return "\n".join(lines)


def build_financial_context(financial_reports: list[dict[str, Any]]) -> str:
    if not financial_reports:
        return "当前没有抓取到财报公告。"

    lines: list[str] = []
    for item in financial_reports[:8]:
        lines.append(f"### {item.get('date', '')} {item.get('title', '')}")
        if item.get("pdf_url"):
            lines.append(f"- PDF：{item['pdf_url']}")
        if item.get("financial_signals"):
            for signal in item["financial_signals"][:6]:
                lines.append(f"- 正文线索：{signal}")
        elif item.get("content_excerpt"):
            lines.append(f"- 正文节选：{item['content_excerpt'][:300]}")
    return "\n".join(lines)


def local_expert_reply(result: dict[str, Any], question: str) -> str:
    portfolio = result["portfolio"]
    market_df: pd.DataFrame = result["market_df"]
    news_items = result["news_items"]
    financial_reports = result["financial_reports"]
    position = portfolio["positions"][0]
    related_risk_news = [item for item in news_items if item.get("sentiment_hint") == "risk"]
    related_positive_news = [item for item in news_items if item.get("sentiment_hint") == "positive"]

    lines = [
        f"## 专家意见：{position['name']}（{position['symbol']}）",
        "",
        f"你的问题：{question}",
        "",
        "### 事实依据",
        f"- 持仓行业：{position.get('sector', '未知')}；成本价：{position.get('cost_price')}；当前价：{position.get('market_price')}。",
        f"- 新闻样本：{len(news_items)} 条，其中风险标签 {len(related_risk_news)} 条、正面标签 {len(related_positive_news)} 条。",
        f"- 财报/公告样本：{len(financial_reports)} 条。",
    ]

    if not market_df.empty:
        symbol = position["symbol"]
        group = market_df[market_df["symbol"] == symbol]
        indicators = calculate_indicators(group, result.get("quote_metrics", {}).get(symbol))
        classes = classify_indicators(indicators)
        lines.extend(
            [
                f"- 行情区间：{indicators['start_date']} 至 {indicators['end_date']}，区间收益 {indicators['period_return_pct']:.2f}%，最大回撤 {indicators['max_drawdown_pct']:.2f}%。",
                f"- MA结构：MA5/MA20/MA60 为 {indicators['ma5']:.2f}/{indicators['ma20']:.2f}/{indicators['ma60']:.2f}，判断为{classes['trend']}。",
                f"- MACD：DIF {indicators['macd_dif']:.2f}、DEA {indicators['macd_dea']:.2f}、柱值 {indicators['macd_bar']:.2f}，判断为{classes['macd']}。",
                f"- 量能：成交量/5日均量为 {indicators['volume_ratio_5']:.2f}，属于{classes['volume']}。",
            ]
        )

    financial_signals = [
        signal
        for report in financial_reports
        for signal in report.get("financial_signals", [])[:3]
    ]
    if financial_signals:
        lines.extend(["", "### 财报正文线索"])
        lines.extend(f"- {signal}" for signal in financial_signals[:8])

    lines.extend(
        [
            "",
            "### 初步判断",
            "- 如果短期价格在主要均线下方，同时 MACD 仍弱，说明技术面修复尚未形成强一致性，需要继续观察价格是否重新站回短中期均线。",
            "- 如果 MACD 柱由负值逐步收窄，即使 DIF/DEA 仍在弱区，也代表下行动能可能减弱；但必须配合成交量和基本面线索交叉验证。",
            "- 成交量放大但价格不涨，通常说明分歧或抛压增加；缩量反弹说明承接不足；放量上涨才更能支持修复信号。",
            "- 财报中的收入、净利润、现金流和主营产品价格变化，是判断技术反弹能否持续的重要约束。",
            "",
            "### 后续观察",
            "- 价格是否站回 MA20，且 MA5 是否上穿 MA20。",
            "- MACD 是否形成 DIF 上穿 DEA，并且柱值连续扩张。",
            "- 成交量是否在上涨日同步放大，而非只在下跌日放大。",
            "- 最新财报或业绩说明会是否解释利润变化、产品价格、订单和现金流。",
        ]
    )
    return ensure_disclaimer("\n".join(lines))


def llm_expert_reply(
    result: dict[str, Any],
    question: str,
    config: LLMConfig,
    history: list[dict[str, str]] | None = None,
) -> str:
    context = build_expert_context(result)
    history_text = ""
    if history:
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-8:])
    user_prompt = f"""当前报告上下文如下：

{context}

最近对话：
{history_text}

用户最新问题：
{question}
"""
    return ensure_disclaimer(LLMClient(config).chat(EXPERT_SYSTEM_PROMPT, user_prompt))
