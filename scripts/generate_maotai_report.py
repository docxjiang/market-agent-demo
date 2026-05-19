from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.a_share_fetcher import (
    fetch_a_share_market_data,
    fetch_a_share_news,
    fetch_a_share_quote_metrics,
    fetch_recent_financial_reports,
    ensure_technical_market_data,
    save_json,
)
from src.agents import run_all_agents
from src.data_loader import load_market_data, load_portfolio
from src.charting import generate_technical_charts
from src.news_fetcher import MIN_NEWS_COUNT
from src.report import build_full_news_report
from src.llm_refiner import refine_report_with_llm


def main() -> None:
    portfolio = load_portfolio(PROJECT_ROOT / "data" / "a_share_portfolio.json")
    market_df = fetch_a_share_market_data(portfolio, beg="20250101")
    if market_df.empty:
        market_df = load_market_data(PROJECT_ROOT / "data" / "a_share_market.csv")

    news_items = fetch_a_share_news(portfolio, min_count=MIN_NEWS_COUNT)
    financial_reports = fetch_recent_financial_reports(portfolio)
    quote_metrics = fetch_a_share_quote_metrics(portfolio)
    chart_paths = {}
    chart_dir = PROJECT_ROOT / "outputs" / "charts"
    for position in portfolio.get("positions", []):
        symbol = position["symbol"]
        generated = generate_technical_charts(market_df, symbol, chart_dir)
        chart_paths[symbol] = {
            name: str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for name, path in generated.items()
        }

    save_json(news_items, PROJECT_ROOT / "outputs" / "maotai_news.json")
    save_json(financial_reports, PROJECT_ROOT / "outputs" / "maotai_financial_reports.json")
    (PROJECT_ROOT / "outputs" / "maotai_quote_metrics.json").write_text(
        __import__("json").dumps(quote_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    market_df.to_csv(PROJECT_ROOT / "outputs" / "maotai_market_for_indicators.csv", index=False, encoding="utf-8")

    agent_results = run_all_agents(
        portfolio,
        news_items,
        market_df,
        financial_reports=financial_reports,
        quote_metrics=quote_metrics,
    )
    report = build_full_news_report(
        portfolio,
        market_df,
        news_items,
        agent_results,
        financial_reports=financial_reports,
        chart_paths=chart_paths,
    )
    if os.getenv("USE_LLM_REFINER", "0") == "1":
        report = refine_report_with_llm(report)

    output_path = PROJECT_ROOT / "outputs" / "maotai_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Fetched A-share news count: {len(news_items)}")
    print(f"Fetched financial report/announcement count: {len(financial_reports)}")
    print(f"Fetched quote metrics count: {len(quote_metrics)}")
    print(f"Maotai report generated: {output_path}")


if __name__ == "__main__":
    main()
