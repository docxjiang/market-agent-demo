from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.a_share_fetcher import (
    fetch_a_share_market_data,
    fetch_a_share_news,
    fetch_a_share_quote_metrics,
    fetch_recent_financial_reports,
    save_json,
)
from src.agents import run_all_agents
from src.charting import generate_technical_charts
from src.llm_refiner import refine_report_with_llm
from src.news_fetcher import MIN_NEWS_COUNT
from src.report import build_full_news_report


def infer_exchange(code: str) -> tuple[str, str, str]:
    if code.startswith("6"):
        return f"{code}.SH", "上海证券交易所", f"1.{code}"
    return f"{code}.SZ", "深圳证券交易所", f"0.{code}"


def build_portfolio(args: argparse.Namespace) -> dict:
    symbol, exchange, secid = infer_exchange(args.code)
    return {
        "user_id": "demo_a_share_user_dynamic",
        "base_currency": "CNY",
        "as_of_date": args.as_of_date,
        "risk_profile": "moderate",
        "market": "A股",
        "positions": [
            {
                "symbol": symbol,
                "code": args.code,
                "name": args.name,
                "asset_type": "A股",
                "quantity": args.quantity,
                "cost_price": args.cost_price,
                "market_price": args.market_price,
                "sector": args.sector,
                "exchange": exchange,
                "secid": secid,
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an A-share holding analysis report.")
    parser.add_argument("--code", required=True, help="A-share code, e.g. 000657")
    parser.add_argument("--name", required=True, help="Stock name, e.g. 中钨高新")
    parser.add_argument("--sector", default="A股", help="Industry/sector")
    parser.add_argument("--quantity", type=float, default=1000)
    parser.add_argument("--cost-price", type=float, default=10.0)
    parser.add_argument("--market-price", type=float, default=10.0)
    parser.add_argument("--as-of-date", default="2026-05-17")
    parser.add_argument("--output-prefix", default="", help="Output file prefix. Defaults to code_name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    portfolio = build_portfolio(args)
    symbol = portfolio["positions"][0]["symbol"]
    prefix = args.output_prefix or f"{args.code}_{args.name}"

    market_df = fetch_a_share_market_data(portfolio, beg="20250101")
    quote_metrics = fetch_a_share_quote_metrics(portfolio)
    news_items = fetch_a_share_news(portfolio, min_count=MIN_NEWS_COUNT)
    financial_reports = fetch_recent_financial_reports(portfolio)

    output_dir = PROJECT_ROOT / "outputs"
    save_json(news_items, output_dir / f"{prefix}_news.json")
    save_json(financial_reports, output_dir / f"{prefix}_financial_reports.json")
    (output_dir / f"{prefix}_quote_metrics.json").write_text(
        json.dumps(quote_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not market_df.empty:
        market_df.to_csv(output_dir / f"{prefix}_market_for_indicators.csv", index=False, encoding="utf-8")

    chart_paths = {}
    if not market_df.empty:
        generated = generate_technical_charts(market_df, symbol, output_dir / "charts")
        chart_paths[symbol] = {
            name: str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for name, path in generated.items()
        }

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

    output_path = output_dir / f"{prefix}_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Stock: {args.name} ({symbol})")
    print(f"Fetched market rows: {len(market_df)}")
    print(f"Fetched news count: {len(news_items)}")
    print(f"Fetched financial report/announcement count: {len(financial_reports)}")
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
