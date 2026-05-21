from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.agents import AgentResult
from src.decision_summary import build_decision_summary, decision_summary_to_markdown


DISCLAIMER = "免责声明：本系统仅用于课程演示和市场信息解读，不构成投资建议。"


def build_demo_report() -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# 个人持仓新闻解读与风险提醒报告

生成时间：{generated_at}

## 报告状态

第一阶段示例报告。当前版本暂未调用真实大模型 API。

## 风险提示

- 请关注单一行业或单一标的持仓集中带来的波动。
- 请关注新闻事件与市场情绪变化可能带来的短期价格波动。

{DISCLAIMER}
"""


def build_portfolio_snapshot(portfolio: dict[str, Any]) -> str:
    positions = portfolio.get("positions", [])
    total_value = sum(float(item["quantity"]) * float(item["market_price"]) for item in positions)
    lines = ["## 持仓快照", ""]
    lines.append(f"- 账户基准货币：{portfolio.get('base_currency', '未知')}")
    lines.append(f"- 数据日期：{portfolio.get('as_of_date', '未知')}")
    lines.append(f"- 示例持仓数量：{len(positions)}")
    lines.append(f"- 示例总市值：{total_value:,.2f}")
    return "\n".join(lines)


def build_market_snapshot(market_df: pd.DataFrame) -> str:
    lines = ["## 行情快照", ""]
    for symbol, group in market_df.sort_values("date").groupby("symbol"):
        start = group.iloc[0]
        end = group.iloc[-1]
        change_pct = (float(end["close"]) / float(start["close"]) - 1) * 100
        lines.append(
            f"- {symbol}：{start['date']} 至 {end['date']}，收盘价从 {start['close']} 变为 {end['close']}，变化 {change_pct:.2f}%。"
        )
    return "\n".join(lines)


def build_report(
    portfolio: dict[str, Any],
    market_df: pd.DataFrame,
    agent_results: list[AgentResult],
    news_items: list[dict[str, Any]] | None = None,
    financial_reports: list[dict[str, Any]] | None = None,
    chart_paths: dict[str, dict[str, str]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    decision_summary: dict[str, Any] | None = None,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        "# 个人持仓新闻解读与风险提醒报告",
        "",
        f"生成时间：{generated_at}",
        "",
        "> 本报告由多智能体流程生成；如启用 LLM 精修，会在规则分析结果基础上进行语言组织。",
        "",
        decision_summary_to_markdown(
            decision_summary
            or build_decision_summary(
                portfolio,
                market_df,
                news_items or [],
                financial_reports=financial_reports or [],
                quote_metrics=quote_metrics or {},
            )
        ),
        "",
        build_portfolio_snapshot(portfolio),
        "",
        build_market_snapshot(market_df),
        "",
    ]

    if chart_paths:
        sections.extend([build_chart_section(chart_paths), ""])

    if news_items is not None:
        sections.extend([build_news_snapshot(news_items), ""])
    if financial_reports is not None:
        sections.extend([build_financial_report_snapshot(financial_reports), ""])

    sections.extend(["## 多智能体分析结果", ""])

    for result in agent_results:
        sections.append(result.content)
        sections.append("")

    sections.extend(
        [
            "## 综合提醒",
            "",
            "- 当前示例持仓集中于 A 股单一标的，报告重点应放在业绩增速、分红回购、价格体系、订单/库存、管理层/公告事件和资金面变化。",
            "- 新闻与公告需要交叉验证：新闻负责捕捉市场预期变化，公告负责确认公司正式披露口径，行情负责观察预期是否已经反映到价格和成交量中。",
            "- 后续最值得跟踪的是年度报告和季度报告中的收入、净利润、现金分红、回购进展，以及业绩说明会对经营目标和渠道动销的回应。",
            "- 本系统只做信息整理、现象描述和风险提示，不输出任何买入、卖出或持有建议。",
            "",
            DISCLAIMER,
        ]
    )
    return "\n".join(sections)


def build_news_snapshot(news_items: list[dict[str, Any]]) -> str:
    source_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    sentiment_counts: dict[str, int] = {}

    for item in news_items:
        source = item.get("fetched_via") or item.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        sentiment = item.get("sentiment_hint", "neutral")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        for symbol in item.get("related_symbols", []):
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    lines = ["## 新闻抓取概览", ""]
    lines.append(f"- 本次纳入新闻数量：{len(news_items)} 条")
    lines.append(
        "- 抓取来源类型："
        + "，".join(f"{source} {count} 条" for source, count in sorted(source_counts.items()))
    )
    if symbol_counts:
        lines.append(
            "- 关联标的分布："
            + "，".join(f"{symbol} {count} 条" for symbol, count in sorted(symbol_counts.items()))
        )
    lines.append(
        "- 情绪/风险标签分布："
        + "，".join(f"{label} {count} 条" for label, count in sorted(sentiment_counts.items()))
    )
    return "\n".join(lines)


def build_full_news_report(
    portfolio: dict[str, Any],
    market_df: pd.DataFrame,
    news_items: list[dict[str, Any]],
    agent_results: list[AgentResult],
    financial_reports: list[dict[str, Any]] | None = None,
    chart_paths: dict[str, dict[str, str]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    decision_summary: dict[str, Any] | None = None,
) -> str:
    report = build_report(
        portfolio,
        market_df,
        agent_results,
        news_items=news_items,
        financial_reports=financial_reports,
        chart_paths=chart_paths,
        quote_metrics=quote_metrics,
        decision_summary=decision_summary,
    )
    lines = [report]

    if financial_reports is not None:
        lines.extend(["", "## 最近财报与财务相关公告明细", ""])
        for index, item in enumerate(financial_reports, start=1):
            lines.append(f"### {index}. {item.get('title', '未命名公告')}")
            lines.append("")
            lines.append(f"- 股票：{item.get('name', '')}（{item.get('symbol', '')}）")
            lines.append(f"- 日期：{item.get('date', '未知')}")
            lines.append(f"- 来源：{item.get('source', '未知')}")
            if item.get("columns"):
                lines.append(f"- 栏目：{', '.join(item.get('columns', []))}")
            if item.get("url"):
                lines.append(f"- 链接：{item['url']}")
            lines.append("")

    lines.extend(["", "## 新闻明细（不少于 50 条）", ""])

    for index, item in enumerate(news_items, start=1):
        symbols = ", ".join(item.get("related_symbols", [])) or "未直接匹配持仓标的"
        url = item.get("url", "")
        lines.append(f"### {index}. {item.get('title', '未命名新闻')}")
        lines.append("")
        lines.append(f"- 来源：{item.get('source', '未知')}")
        lines.append(f"- 日期：{item.get('date', '未知')}")
        lines.append(f"- 相关标的：{symbols}")
        lines.append(f"- 风险标签：{item.get('sentiment_hint', 'neutral')}")
        lines.append(f"- 抓取方式：{item.get('fetched_via', 'unknown')}")
        if url:
            lines.append(f"- 链接：{url}")
        if item.get("summary"):
            lines.append(f"- 摘要：{item['summary']}")
        lines.append("")

    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_financial_report_snapshot(financial_reports: list[dict[str, Any]]) -> str:
    lines = ["## 最近财报与公告概览", ""]
    lines.append(f"- 本次纳入财报/财务相关公告数量：{len(financial_reports)} 条")
    if financial_reports:
        latest = financial_reports[0]
        lines.append(
            f"- 最新一条：{latest.get('date', '未知日期')} {latest.get('title', '未命名公告')}"
        )
    return "\n".join(lines)


def build_chart_section(chart_paths: dict[str, dict[str, str]]) -> str:
    lines = ["## 技术图表", ""]
    for symbol, paths in chart_paths.items():
        lines.append(f"### {symbol}")
        if paths.get("kline_ma"):
            lines.append("K 线叠加 MA5 / MA10 / MA20：")
            lines.append(f"![{symbol} K线均线图]({paths['kline_ma']})")
            lines.append("")
        if paths.get("macd"):
            lines.append("MACD：")
            lines.append(f"![{symbol} MACD图]({paths['macd']})")
            lines.append("")
        if paths.get("amount"):
            lines.append("成交额（若数据源未直接提供成交额，则按均价×成交量估算）：")
            lines.append(f"![{symbol} 成交额图]({paths['amount']})")
            lines.append("")
    return "\n".join(lines).rstrip()
