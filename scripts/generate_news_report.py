from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import run_all_agents
from src.data_loader import load_market_data, load_portfolio
from src.news_fetcher import MIN_NEWS_COUNT, fetch_portfolio_news, save_news
from src.report import build_full_news_report


def main() -> None:
    portfolio = load_portfolio(PROJECT_ROOT / "data" / "sample_portfolio.json")
    market_df = load_market_data(PROJECT_ROOT / "data" / "sample_market.csv")
    news_items = fetch_portfolio_news(portfolio, min_count=MIN_NEWS_COUNT)
    save_news(news_items, PROJECT_ROOT / "outputs" / "latest_news.json")

    agent_results = run_all_agents(portfolio, news_items, market_df)
    report = build_full_news_report(portfolio, market_df, news_items, agent_results)

    output_path = PROJECT_ROOT / "outputs" / "news_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Fetched news count: {len(news_items)}")
    print(f"News report generated: {output_path}")


if __name__ == "__main__":
    main()
