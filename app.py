from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import logging
from datetime import date
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agents import (
    run_all_agents,
    run_financial_report_agent,
    run_news_agent,
    run_risk_agent,
    run_technical_agent,
)
from src.a_share_fetcher import (
    fetch_a_share_market_data,
    fetch_a_share_intraday_market_data,
    fetch_a_share_news,
    fetch_a_share_quote_metrics,
    fetch_recent_financial_reports,
    resolve_stock_profile,
    save_json,
)
from src.charting import generate_technical_charts
from src.decision_summary import build_decision_summary
from src.expert import llm_expert_reply, local_expert_reply
from src.interactive_charting import (
    CHART_WINDOW_OPTIONS,
    build_chart_summary_html,
    build_interactive_technical_figure,
    build_key_metrics_table,
    build_period_return_table,
    format_chart_range,
    format_data_source_label,
    prepare_interactive_price_frame,
    select_chart_window_frame,
)
from src.llm_client import LLMClient, LLMConfig, llm_timeout_seconds
from src.news_fetcher import MIN_NEWS_COUNT
from src.report import build_full_news_report
from src.ui_components import (
    build_agent_analysis_html,
    build_announcement_cards_html,
    build_decision_summary_html,
    build_news_cards_html,
)


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DISCLAIMER = "免责声明：本系统仅用于课程演示和市场信息解读，不构成投资建议。"
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

PROGRESS_STEPS = [
    "识别股票与持仓",
    "获取行情并展示K线图",
    "搜索行业新闻",
    "获取财务报告",
    "获取实时行情指标",
    "等待LLM智能体反馈",
    "AI信息整合",
    "保存报告与数据",
]

CHART_PREVIEW_TITLE = "技术图表(预览)"
LLM_WAIT_MESSAGES = [
    "金融分析师正在读取财报公告。",
    "正在检查历史交易行情。",
    "正在对照新闻事件与价格反应。",
    "正在梳理资金面和波动风险。",
    "正在识别短线催化与中期变量。",
    "正在核对关键财务线索。",
    "正在过滤弱相关信息。",
    "正在生成多智能体交叉判断。",
    "正在压缩规则结果并保留关键证据。",
    "正在形成综合判断。",
]


def format_progress_step(step: int, use_llm: bool = True) -> str:
    bounded = min(max(step, 1), len(PROGRESS_STEPS))
    label = PROGRESS_STEPS[bounded - 1]
    if bounded == 6 and not use_llm:
        label = "运行本地规则智能体"
    return f"{label}（{bounded}/{len(PROGRESS_STEPS)}）"


def llm_max_parallel_requests(default: int = 2) -> int:
    try:
        return max(1, int(os.getenv("LLM_MAX_PARALLEL_REQUESTS", str(default))))
    except ValueError:
        return default


def infer_exchange(code: str) -> tuple[str, str, str]:
    code = code.strip()
    if code.startswith("6"):
        return f"{code}.SH", "上海证券交易所", f"1.{code}"
    return f"{code}.SZ", "深圳证券交易所", f"0.{code}"


def load_stock_pool() -> list[dict[str, Any]]:
    path = BASE_DIR / "data" / "a_share_stock_pool.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_stock_preview(code: str, stock_pool: list[dict[str, Any]]) -> dict[str, str]:
    code = code.strip()
    if len(code) != 6 or not code.isdigit():
        return {}

    symbol, exchange, _ = infer_exchange(code)
    matched = next(
        (
            item
            for item in stock_pool
            if str(item.get("code", "")).zfill(6) == code
            or str(item.get("symbol", "")).split(".")[0].zfill(6) == code
        ),
        {},
    )
    name = str(matched.get("name") or "")
    sector = str(matched.get("sector") or "")
    return {
        "code": code,
        "symbol": symbol,
        "name": "" if name == code else name,
        "sector": "" if sector in {"A股", "沪深A股", "股票"} else sector,
        "exchange": str(matched.get("exchange") or exchange),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def resolve_stock_preview_cached(code: str) -> dict[str, str]:
    if len(code.strip()) != 6 or not code.strip().isdigit():
        return {}
    try:
        profile = resolve_stock_profile(code.strip(), load_stock_pool())
    except Exception:
        return lookup_stock_preview(code, load_stock_pool())

    name = str(profile.get("name") or "")
    sector = str(profile.get("sector") or "")
    return {
        "code": str(profile.get("code") or code).strip(),
        "symbol": str(profile.get("symbol") or infer_exchange(code)[0]),
        "name": "" if name == code else name,
        "sector": "" if sector in {"A股", "沪深A股", "股票", "未识别行业"} else sector,
        "exchange": str(profile.get("exchange") or infer_exchange(code)[1]),
    }


def run_with_llm_wait_messages(
    worker: Callable[[], Any],
    wait_callback: Callable[[str], None] | None,
    messages: list[str] | None = None,
    interval_seconds: float = 5.0,
) -> Any:
    if wait_callback is None:
        return worker()

    wait_messages = messages or LLM_WAIT_MESSAGES
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        index = 0
        while True:
            wait_callback(wait_messages[index % len(wait_messages)])
            index += 1
            try:
                return future.result(timeout=interval_seconds)
            except FutureTimeoutError:
                continue


def run_ai_analysis_parallel(
    portfolio: dict[str, Any],
    news_items: list[dict[str, Any]],
    market_df: pd.DataFrame,
    financial_reports: list[dict[str, Any]] | None = None,
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    if not use_llm:
        agent_results = run_all_agents(
            portfolio,
            news_items,
            market_df,
            financial_reports=financial_reports,
            quote_metrics=quote_metrics,
            llm_client=False,
        )
        decision_summary = build_decision_summary(
            portfolio,
            market_df,
            news_items,
            financial_reports=financial_reports,
            quote_metrics=quote_metrics,
            prefer_llm=False,
        )
        return agent_results, decision_summary

    with ThreadPoolExecutor(max_workers=llm_max_parallel_requests()) as executor:
        news_future = executor.submit(run_news_agent, portfolio, news_items, llm_client)
        technical_future = executor.submit(
            run_technical_agent,
            market_df,
            quote_metrics,
            llm_client,
        )
        risk_future = executor.submit(
            run_risk_agent,
            portfolio,
            news_items,
            market_df,
            financial_reports,
            quote_metrics,
            llm_client,
        )
        summary_future = executor.submit(
            build_decision_summary,
            portfolio,
            market_df,
            news_items,
            financial_reports=financial_reports,
            quote_metrics=quote_metrics,
            llm_client=llm_client,
            prefer_llm=True,
        )
        agent_results = [news_future.result()]
        if financial_reports is not None:
            agent_results.append(run_financial_report_agent(financial_reports, news_items=news_items))
        agent_results.extend([technical_future.result(), risk_future.result()])
        return agent_results, summary_future.result()


def should_use_llm_analysis(expert_config: dict[str, Any]) -> bool:
    return expert_config.get("mode") != "本地规则专家" and bool(expert_config.get("api_key"))


def build_analysis_llm_client(expert_config: dict[str, Any]) -> LLMClient | None:
    if not should_use_llm_analysis(expert_config):
        return None
    config = LLMConfig(
        api_key=str(expert_config["api_key"]),
        base_url=str(expert_config["base_url"]),
        model=str(expert_config["model"]),
        timeout_seconds=llm_timeout_seconds(),
        temperature=float(expert_config["temperature"]),
    )
    return LLMClient(config)


def build_portfolio(profile: dict[str, Any], quantity: float, cost_price: float) -> dict[str, Any]:
    return build_portfolio_from_positions(
        [{"code": profile["code"], "quantity": quantity, "cost_price": cost_price}],
        [profile],
    )


def normalize_positions_input(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    normalized: list[dict[str, float | str]] = []
    seen_codes: set[str] = set()
    for row in rows:
        code = str(row.get("股票代码") or row.get("code") or "").strip().zfill(6)
        if len(code) != 6 or not code.isdigit() or code in seen_codes:
            continue
        quantity = _safe_float(row.get("持仓数量", row.get("quantity", 0)))
        cost_price = _safe_float(row.get("成本价", row.get("cost_price", 0)))
        normalized.append({"code": code, "quantity": quantity, "cost_price": cost_price})
        seen_codes.add(code)
    return normalized


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_portfolio_from_positions(
    positions_input: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    profile_by_code = {str(profile.get("code", "")): profile for profile in profiles}
    positions: list[dict[str, Any]] = []
    for item in positions_input:
        code = str(item["code"])
        profile = profile_by_code[code]
        market_price = float(profile.get("latest_price") or item.get("cost_price") or 0)
        positions.append(
            {
                "symbol": profile["symbol"],
                "code": profile["code"],
                "name": profile["name"],
                "asset_type": "A股",
                "quantity": float(item.get("quantity", 0)),
                "cost_price": float(item.get("cost_price", 0)),
                "market_price": market_price,
                "sector": profile["sector"],
                "exchange": profile["exchange"],
                "secid": profile["secid"],
            }
        )
    return {
        "user_id": "streamlit_user",
        "base_currency": "CNY",
        "as_of_date": date.today().isoformat(),
        "risk_profile": "moderate",
        "market": "A股",
        "positions": positions,
    }


def output_prefix(code: str, name: str) -> str:
    safe_name = "".join(ch for ch in name if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return f"{code}_{safe_name or 'stock'}"


def portfolio_output_prefix(portfolio: dict[str, Any]) -> str:
    positions = portfolio.get("positions", [])
    if len(positions) == 1:
        position = positions[0]
        return output_prefix(str(position.get("code", "stock")), str(position.get("name", "stock")))
    codes = "_".join(str(position.get("code", "")) for position in positions[:3])
    return f"portfolio_{len(positions)}_{codes or 'stocks'}"


def positions_input_from_portfolio(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "code": str(position.get("code", "")),
            "quantity": float(position.get("quantity", 0)),
            "cost_price": float(position.get("cost_price", 0)),
        }
        for position in portfolio.get("positions", [])
    ]


def related_news_for_portfolio(news_items: list[dict[str, Any]], portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    symbols = {position.get("symbol") for position in portfolio.get("positions", []) if position.get("symbol")}
    related = [
        item
        for item in news_items
        if symbols.intersection(set(item.get("related_symbols", [])))
    ]
    return related or news_items


def dedupe_items_by_key(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get("url") or item.get("id") or item.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def filter_items_for_symbol(items: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in items:
        direct_symbol = str(item.get("symbol") or "")
        related_symbols = {str(related) for related in item.get("related_symbols", [])}
        if direct_symbol == symbol or symbol in related_symbols:
            selected.append(item)
    return selected


def position_label(position: dict[str, Any]) -> str:
    name = str(position.get("name") or "")
    symbol = str(position.get("symbol") or "")
    return f"{name}（{symbol}）" if name else symbol


def find_agent_result(agent_results: list[Any], name_fragment: str) -> Any | None:
    return next((item for item in agent_results if name_fragment in getattr(item, "name", "")), None)


def technical_agent_section_for_label(content: str, label: str) -> str:
    heading_pattern = re.compile(r"^### .+（[0-9]{6}\.(?:SH|SZ)）\s*$|^### [0-9]{6}\.(?:SH|SZ)\s*$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(content))
    head = content[: matches[0].start()].strip() if matches else content.strip()
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section = content[match.start() : next_start].strip()
        first_line = section.splitlines()[0] if section.splitlines() else ""
        if label in first_line:
            return "\n\n".join([head, section]).strip()
    return content


def fetch_news_for_positions(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    positions = portfolio.get("positions", [])
    if len(positions) <= 1:
        return fetch_a_share_news(portfolio, min_count=MIN_NEWS_COUNT)
    per_position = max(6, MIN_NEWS_COUNT // max(1, len(positions)))
    items: list[dict[str, Any]] = []
    for position in positions:
        single_portfolio = {**portfolio, "positions": [position]}
        items.extend(fetch_a_share_news(single_portfolio, min_count=per_position))
    return dedupe_items_by_key(items)


@st.cache_data(show_spinner=False, ttl=600)
def generate_report_cached(
    code: str,
    quantity: float,
    cost_price: float,
) -> dict[str, Any]:
    return generate_report(
        code=code,
        quantity=quantity,
        cost_price=cost_price,
    )


def generate_report(
    code: str | None = None,
    quantity: float = 0.0,
    cost_price: float = 0.0,
    positions_input: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    chart_preview_callback: Callable[[dict[str, Any]], None] | None = None,
    llm_wait_callback: Callable[[str], None] | None = None,
    use_llm_analysis: bool = True,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    def progress(step: int) -> None:
        if progress_callback:
            progress_callback(step)

    progress(1)
    normalized_positions = normalize_positions_input(
        positions_input
        or [{"code": code or "", "quantity": quantity, "cost_price": cost_price}]
    )
    if not normalized_positions:
        raise ValueError("至少需要一个有效的 6 位 A 股代码。")
    profiles = [resolve_stock_profile(str(item["code"]), load_stock_pool()) for item in normalized_positions]
    portfolio = build_portfolio_from_positions(normalized_positions, profiles)
    profile = profiles[0]
    prefix = portfolio_output_prefix(portfolio)

    progress(2)
    market_frames: list[pd.DataFrame] = []
    intraday_frames: list[pd.DataFrame] = []
    chart_paths: dict[str, dict[str, str]] = {}
    chart_abs_paths: dict[str, dict[str, str]] = {}
    preview_snapshots: list[dict[str, Any]] = []

    for index, position in enumerate(portfolio["positions"]):
        symbol = str(position["symbol"])
        single_portfolio = {**portfolio, "positions": [position]}
        position_market_df = fetch_a_share_market_data(single_portfolio, beg="20250101")
        position_intraday_df = fetch_a_share_intraday_market_data(single_portfolio, frequencies=("5", "60"))
        if not position_market_df.empty:
            latest_close = float(position_market_df.sort_values("date").iloc[-1]["close"])
            position["market_price"] = latest_close
            profiles[index]["latest_price"] = latest_close
            generated = generate_technical_charts(position_market_df, symbol, OUTPUT_DIR / "charts")
            chart_paths[symbol] = {
                key: str(path.relative_to(BASE_DIR)).replace("\\", "/")
                for key, path in generated.items()
            }
            chart_abs_paths[symbol] = {key: str(path) for key, path in generated.items()}
            market_frames.append(position_market_df)
        if not position_intraday_df.empty:
            intraday_frames.append(position_intraday_df)
        preview_snapshots.append(
            {
                "label": position_label(position),
                "profile": profiles[index],
                "market_df": position_market_df,
                "intraday_market_df": position_intraday_df,
            }
        )
        if chart_preview_callback:
            chart_preview_callback(
                {
                    "portfolio": {**portfolio, "positions": portfolio["positions"][: index + 1]},
                    "profile": profiles[index],
                    "market_df": position_market_df,
                    "intraday_market_df": position_intraday_df,
                    "snapshots": preview_snapshots.copy(),
                    "completed": index + 1,
                    "total": len(portfolio["positions"]),
                }
            )

    market_df = (
        pd.concat(market_frames, ignore_index=True).sort_values(["symbol", "date"])
        if market_frames
        else pd.DataFrame()
    )
    intraday_market_df = (
        pd.concat(intraday_frames, ignore_index=True).sort_values(["symbol", "date"])
        if intraday_frames
        else pd.DataFrame()
    )

    progress(3)
    news_items = fetch_news_for_positions(portfolio)
    progress(4)
    financial_reports = fetch_recent_financial_reports(portfolio)
    progress(5)
    quote_metrics = fetch_a_share_quote_metrics(portfolio)

    progress(6)
    if use_llm_analysis:
        agent_results, decision_summary = run_with_llm_wait_messages(
            lambda: run_ai_analysis_parallel(
                portfolio,
                news_items,
                market_df,
                financial_reports=financial_reports,
                quote_metrics=quote_metrics,
                use_llm=True,
                llm_client=llm_client,
            ),
            llm_wait_callback,
        )
    else:
        agent_results, decision_summary = run_ai_analysis_parallel(
            portfolio,
            news_items,
            market_df,
            financial_reports=financial_reports,
            quote_metrics=quote_metrics,
            use_llm=False,
        )
    progress(7)
    report = build_full_news_report(
        portfolio,
        market_df,
        news_items,
        agent_results,
        financial_reports=financial_reports,
        chart_paths=chart_paths,
        quote_metrics=quote_metrics,
        decision_summary=decision_summary,
    )

    progress(8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(news_items, OUTPUT_DIR / f"{prefix}_news.json")
    save_json(financial_reports, OUTPUT_DIR / f"{prefix}_financial_reports.json")
    (OUTPUT_DIR / f"{prefix}_quote_metrics.json").write_text(
        json.dumps(quote_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not market_df.empty:
        market_df.to_csv(OUTPUT_DIR / f"{prefix}_market_for_indicators.csv", index=False, encoding="utf-8")
    if not intraday_market_df.empty:
        intraday_market_df.to_csv(OUTPUT_DIR / f"{prefix}_intraday_market_for_chart.csv", index=False, encoding="utf-8")
    report_path = OUTPUT_DIR / f"{prefix}_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "portfolio": portfolio,
        "profile": profile,
        "profiles": profiles,
        "market_df": market_df,
        "intraday_market_df": intraday_market_df,
        "quote_metrics": quote_metrics,
        "news_items": news_items,
        "financial_reports": financial_reports,
        "agent_results": agent_results,
        "decision_summary": decision_summary,
        "report": report,
        "report_path": str(report_path),
        "chart_abs_paths": chart_abs_paths,
        "use_llm_analysis": use_llm_analysis,
        "output_prefix": prefix,
    }


def sidebar_inputs() -> dict[str, Any]:
    st.sidebar.header("股票与持仓")
    preset = st.sidebar.selectbox(
        "示例标的",
        ["中钨高新 000657", "贵州茅台 600519", "多持仓示例", "自定义"],
        index=0,
    )
    defaults = {
        "中钨高新 000657": [{"股票代码": "000657", "持仓数量": 1000.0, "成本价": 50.0}],
        "贵州茅台 600519": [{"股票代码": "600519", "持仓数量": 100.0, "成本价": 1500.0}],
        "多持仓示例": [
            {"股票代码": "000657", "持仓数量": 1000.0, "成本价": 50.0},
            {"股票代码": "600519", "持仓数量": 100.0, "成本价": 1500.0},
        ],
        "自定义": [{"股票代码": "000657", "持仓数量": 1000.0, "成本价": 10.0}],
    }
    edited_rows = st.sidebar.data_editor(
        pd.DataFrame(defaults[preset]),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "股票代码": st.column_config.TextColumn("股票代码", max_chars=6, help="输入 6 位 A 股代码"),
            "持仓数量": st.column_config.NumberColumn("持仓数量", min_value=0.0, step=100.0),
            "成本价": st.column_config.NumberColumn("成本价", min_value=0.0, step=0.01, format="%.2f"),
        },
    )
    positions_input = normalize_positions_input(edited_rows.to_dict("records"))
    previews = [
        resolve_stock_preview_cached(str(item["code"]))
        for item in positions_input
        if len(str(item["code"])) == 6
    ]
    if previews:
        preview_lines = []
        for preview in previews:
            name = preview.get("name") or "未识别股票名称"
            details = f"{preview.get('symbol', '')} · {preview.get('exchange', '')}".strip(" ·")
            if preview.get("sector"):
                details = f"{details} · {preview['sector']}" if details else preview["sector"]
            preview_lines.append(f"**{name}**  \n{details}")
        st.sidebar.markdown("\n\n".join(preview_lines))
    st.sidebar.caption("可输入 1-5 个 A 股代码。股票名称、行业、当前价、交易所会自动解析。")
    first = positions_input[0] if positions_input else {"code": "", "quantity": 0.0, "cost_price": 0.0}

    return {
        "code": str(first["code"]),
        "quantity": float(first["quantity"]),
        "cost_price": float(first["cost_price"]),
        "positions_input": positions_input[:5],
    }


def sidebar_expert_config() -> dict[str, Any]:
    st.sidebar.header("专家模型")
    mode = st.sidebar.radio(
        "专家来源",
        ["当前默认 AI 专家", "API 模型专家", "本地规则专家"],
        index=0,
        horizontal=False,
    )
    config = {
        "mode": mode,
        "base_url": "",
        "api_key": "",
        "model": "",
        "temperature": 0.2,
    }
    if mode == "API 模型专家":
        config["base_url"] = st.sidebar.text_input("Base URL", value="https://api.deepseek.com")
        config["model"] = st.sidebar.text_input("模型名称", value="deepseek-v4-flash")
        config["api_key"] = st.sidebar.text_input("API Key", type="password")
        config["temperature"] = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        st.sidebar.caption("兼容 OpenAI Chat Completions 格式。API Key 只保存在当前页面会话中。")
    elif mode == "当前默认 AI 专家":
        config["base_url"] = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        config["model"] = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
        config["api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        config["temperature"] = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        if config["api_key"]:
            st.sidebar.caption(f"使用环境变量中的默认模型：{config['model']}")
        else:
            st.sidebar.caption("未发现默认 API Key，将自动使用本地规则专家。")
    else:
        st.sidebar.caption("默认使用本地规则专家，不调用外部 API。")
    return config


def render_loading_chart_preview(partial_result: dict[str, Any]) -> None:
    completed = partial_result.get("completed")
    total = partial_result.get("total")
    suffix = f"（{completed}/{total}）" if completed and total else ""
    st.subheader(f"{CHART_PREVIEW_TITLE}{suffix}")
    st.caption("新闻、财报和 LLM 分析仍在后台继续处理。")

    snapshots = partial_result.get("snapshots") or []
    if snapshots:
        visible_snapshots = [
            item
            for item in snapshots
            if not item.get("market_df", pd.DataFrame()).empty
        ]
        if not visible_snapshots:
            st.warning("行情数据暂未加载完成，暂无 K 线预览。")
            return
        tabs = st.tabs([str(item["label"]) for item in visible_snapshots])
        for index, (tab, item) in enumerate(zip(tabs, visible_snapshots)):
            with tab:
                _render_single_loading_chart(
                    item.get("market_df", pd.DataFrame()),
                    item.get("intraday_market_df", pd.DataFrame()),
                    item.get("profile", {}),
                    key_suffix=f"{completed or len(visible_snapshots)}_{total or len(visible_snapshots)}_{index}",
                )
        return

    _render_single_loading_chart(
        partial_result.get("market_df", pd.DataFrame()),
        partial_result.get("intraday_market_df", pd.DataFrame()),
        partial_result.get("profile", {}),
        key_suffix=f"{completed or 1}_{total or 1}_single",
    )


def _render_single_loading_chart(
    market_df: pd.DataFrame,
    intraday_market_df: pd.DataFrame,
    profile: dict[str, Any],
    key_suffix: str,
) -> None:
    if market_df.empty:
        st.warning("行情数据暂未加载完成，暂无 K 线预览。")
        return

    symbol = str(market_df.iloc[-1]["symbol"])
    frame = prepare_interactive_price_frame(market_df, symbol)
    intraday_frame = prepare_interactive_price_frame(intraday_market_df, symbol)
    chart_frame = select_chart_window_frame(frame, intraday_frame, "6个月")
    if chart_frame.empty:
        st.warning("行情数据不足，暂无法生成 K 线预览。")
        return

    latest_close = float(frame.iloc[-1]["close"])
    previous_close = float(frame.iloc[-2]["close"]) if len(frame) > 1 else latest_close
    change = latest_close - previous_close
    change_pct = change / previous_close * 100 if previous_close else 0.0
    chart_source = (
        str(chart_frame.get("data_source", pd.Series(["unknown"])).iloc[-1])
        if "data_source" in chart_frame.columns and not chart_frame.empty
        else "unknown"
    )

    st.html(
        build_chart_summary_html(
            latest_close=latest_close,
            change=change,
            change_pct=change_pct,
            chart_range=format_chart_range(chart_frame),
            volume=float(chart_frame.iloc[-1]["volume"]),
            data_source=format_data_source_label(chart_source),
        )
    )
    st.plotly_chart(
        build_interactive_technical_figure(chart_frame, symbol=symbol, name=str(profile.get("name", ""))),
        width="stretch",
        key=f"loading_preview_chart_{symbol}_{key_suffix}",
        config={"displayModeBar": False, "responsive": True},
    )


def fade_out_loading_preview(preview_box: Any) -> None:
    with preview_box.container():
        st.html(
            """
<style>
.preview-fadeout {
  margin: 8px 0 12px 0;
  padding: 12px 14px;
  border: 1px solid #dbe4ef;
  border-radius: 8px;
  background: #ffffff;
  color: #374151;
  animation: previewFade 650ms ease-out forwards;
}
@keyframes previewFade {
  from { opacity: 1; transform: translateY(0); }
  to { opacity: 0; transform: translateY(-6px); }
}
</style>
<div class="preview-fadeout">正在切换到正式分析界面。</div>
"""
        )
    time.sleep(0.65)
    preview_box.empty()


def render_technical_dashboard(result: dict[str, Any], expert_config: dict[str, Any]) -> None:
    market_df: pd.DataFrame = result["market_df"]
    intraday_market_df: pd.DataFrame = result.get("intraday_market_df", pd.DataFrame())
    portfolio = result["portfolio"]
    quote_metrics = result.get("quote_metrics", {})

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader("交互式技术图表")
    with header_right:
        if st.button("刷新行情和图表", width="stretch"):
            st.cache_data.clear()
            status_box = st.empty()
            progress_bar = st.progress(0)
            preview_box = st.empty()
            use_llm_analysis = should_use_llm_analysis(expert_config)
            analysis_llm_client = build_analysis_llm_client(expert_config)

            def update_progress(step: int) -> None:
                status_box.info(format_progress_step(step, use_llm=use_llm_analysis))
                progress_bar.progress(step / len(PROGRESS_STEPS))

            def update_preview(partial_result: dict[str, Any]) -> None:
                preview_box.empty()
                with preview_box.container():
                    render_loading_chart_preview(partial_result)

            def update_llm_wait(message: str) -> None:
                status_box.info(f"{format_progress_step(6, use_llm=use_llm_analysis)}\n\n{message}")

            st.session_state.latest_result = generate_report(
                positions_input=positions_input_from_portfolio(portfolio),
                progress_callback=update_progress,
                chart_preview_callback=update_preview,
                llm_wait_callback=update_llm_wait if use_llm_analysis else None,
                use_llm_analysis=use_llm_analysis,
                llm_client=analysis_llm_client,
            )
            fade_out_loading_preview(preview_box)
            st.rerun()

    if market_df.empty:
        st.warning("未生成技术图表：行情数据不足或数据源暂不可用。")
        return

    position_by_symbol = {str(item.get("symbol")): item for item in portfolio.get("positions", [])}
    available_symbols = [symbol for symbol in position_by_symbol if symbol in set(market_df["symbol"])]
    if not available_symbols:
        st.warning("暂无可展示的标的行情。")
        return
    selected_label = st.radio(
        "选择标的",
        [position_label(position_by_symbol[symbol]) for symbol in available_symbols],
        horizontal=True,
        key="technical_symbol_selector",
    )
    label_to_symbol = {position_label(position_by_symbol[symbol]): symbol for symbol in available_symbols}
    symbol = label_to_symbol[selected_label]
    group = market_df[market_df["symbol"] == symbol]
    frame = prepare_interactive_price_frame(market_df, symbol)
    if frame.empty:
        st.warning(f"{symbol} 行情数据不足，无法生成交互式图表。")
        return
    intraday_frame = prepare_interactive_price_frame(intraday_market_df, symbol)

    quote = quote_metrics.get(symbol, {})
    name = position_by_symbol.get(symbol, {}).get("name", "")
    latest_close = float(frame.iloc[-1]["close"])
    previous_close = float(frame.iloc[-2]["close"]) if len(frame) > 1 else latest_close
    change = latest_close - previous_close
    change_pct = change / previous_close * 100 if previous_close else 0.0
    source = group.get("data_source", pd.Series(["unknown"])).iloc[-1]

    window = st.radio(
        "时间范围",
        CHART_WINDOW_OPTIONS,
        index=4,
        horizontal=True,
        key=f"chart_window_{symbol}",
    )
    chart_frame = select_chart_window_frame(frame, intraday_frame, window)
    chart_source = (
        str(chart_frame.get("data_source", pd.Series([source])).iloc[-1])
        if "data_source" in chart_frame.columns and not chart_frame.empty
        else str(source)
    )
    if window in {"1日", "1周"} and intraday_frame.empty:
        st.caption("分钟/60分钟 K 线接口暂不可用，短周期视图已回退到日线数据。")

    st.html(
        build_chart_summary_html(
            latest_close=latest_close,
            change=change,
            change_pct=change_pct,
            chart_range=format_chart_range(chart_frame),
            volume=float(chart_frame.iloc[-1]["volume"]),
            data_source=format_data_source_label(chart_source),
        )
    )

    figure = build_interactive_technical_figure(chart_frame, symbol=symbol, name=str(name))
    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": True,
            "scrollZoom": False,
            "responsive": True,
            "modeBarButtonsToRemove": [
                "zoom2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "select2d",
                "lasso2d",
            ],
        },
    )

    period_table = build_period_return_table(frame)
    st.subheader("区间涨跌幅")
    st.dataframe(
        period_table.drop(columns=["涨跌幅数值"]),
        hide_index=True,
        width="stretch",
        column_config={
            "区间": st.column_config.TextColumn("区间"),
            "涨跌幅": st.column_config.TextColumn("涨跌幅"),
        },
    )

    st.subheader("关键行情与技术指标")
    metrics_table = build_key_metrics_table(frame, quote)
    chunk_size = max(1, (len(metrics_table) + 2) // 3)
    metric_cols = st.columns(3)
    for index, col in enumerate(metric_cols):
        start = index * chunk_size
        end = start + chunk_size
        col.dataframe(
            metrics_table.iloc[start:end],
            hide_index=True,
            width="stretch",
            column_config={
                "指标": st.column_config.TextColumn("指标"),
                "数值": st.column_config.TextColumn("数值"),
            },
        )


def render_expert_chat(result: dict[str, Any], expert_config: dict[str, Any]) -> None:
    profile = result["profile"]
    chat_key = f"expert_messages_{profile.get('symbol', 'stock')}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "content": (
                    f"我是当前报告的专家助手，已读取 {profile.get('name', '')}"
                    " 的行情、新闻、财报公告正文线索和多智能体报告。"
                    "你可以问我技术走势、财报压力、新闻风险或需要继续观察的变量。"
                ),
            }
        ]

    left, right = st.columns([3, 1])
    with left:
        st.subheader("专家意见")
    with right:
        if st.button("清空专家对话", width="stretch"):
            st.session_state[chat_key] = []
            st.rerun()

    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("向专家提问，例如：MACD 走弱但成交量放大说明什么？")
    if not prompt:
        return

    st.session_state[chat_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("专家正在读取当前报告并分析..."):
            try:
                if expert_config["mode"] in {"当前默认 AI 专家", "API 模型专家"}:
                    if not expert_config["api_key"]:
                        reply = local_expert_reply(result, prompt)
                    else:
                        config = LLMConfig(
                            api_key=expert_config["api_key"],
                            base_url=expert_config["base_url"],
                            model=expert_config["model"],
                            timeout_seconds=llm_timeout_seconds(),
                            temperature=float(expert_config["temperature"]),
                        )
                        reply = llm_expert_reply(
                            result,
                            prompt,
                            config,
                            history=st.session_state[chat_key],
                        )
                else:
                    reply = local_expert_reply(result, prompt)
            except Exception as exc:
                reply = f"专家模型调用失败：{exc}\n\n已保留当前报告数据，可切换为本地规则专家继续提问。"
            st.markdown(reply)
    st.session_state[chat_key].append({"role": "assistant", "content": reply})


def main() -> None:
    st.set_page_config(
        page_title="A股持仓新闻解读与风险提醒系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.html(
        """
<style>
[data-testid="stSidebar"] {
  min-width: 420px;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] {
  width: 100%;
}
</style>
"""
    )

    st.title("A股个人持仓新闻解读与风险提醒系统")
    st.caption("输入任意 A 股标的，自动抓取新闻、财报公告、行情数据，生成多智能体分析报告。")

    inputs = sidebar_inputs()
    expert_config = sidebar_expert_config()
    code = inputs["code"]
    positions_input = inputs.get("positions_input", [])

    if not positions_input:
        st.error("请在左侧至少输入 1 个有效的 6 位 A 股代码。")
        return

    generate = st.sidebar.button("生成分析报告", type="primary", width="stretch")
    clear_cache = st.sidebar.button("清除缓存后重新抓取", width="stretch")
    if clear_cache:
        st.cache_data.clear()
        st.rerun()

    st.info(DISCLAIMER)

    if not generate and "latest_result" not in st.session_state:
        st.markdown(
            """
            使用方式：
            1. 在左侧输入 6 位 A 股股票代码。
            2. 填写持仓数量和成本价。
            2. 点击“生成分析报告”。
            3. 等待系统抓取新闻、财报公告和行情数据。
            4. 在页面中查看图表、智能体分析和完整报告，也可以下载 Markdown。
            """
        )
        return

    if generate:
        status_box = st.empty()
        progress_bar = st.progress(0)
        preview_box = st.empty()
        use_llm_analysis = should_use_llm_analysis(expert_config)
        analysis_llm_client = build_analysis_llm_client(expert_config)

        def update_progress(step: int) -> None:
            status_box.info(format_progress_step(step, use_llm=use_llm_analysis))
            progress_bar.progress(step / len(PROGRESS_STEPS))

        def update_preview(partial_result: dict[str, Any]) -> None:
            preview_box.empty()
            with preview_box.container():
                render_loading_chart_preview(partial_result)

        def update_llm_wait(message: str) -> None:
            status_box.info(f"{format_progress_step(6, use_llm=use_llm_analysis)}\n\n{message}")

        st.session_state.latest_result = generate_report(
            **inputs,
            progress_callback=update_progress,
            chart_preview_callback=update_preview,
            llm_wait_callback=update_llm_wait if use_llm_analysis else None,
            use_llm_analysis=use_llm_analysis,
            llm_client=analysis_llm_client,
        )
        fade_out_loading_preview(preview_box)
        status_box.success("分析报告生成完成")

    result = st.session_state.latest_result
    portfolio = result["portfolio"]
    profile = result["profile"]
    profiles = result.get("profiles", [profile])
    market_df: pd.DataFrame = result["market_df"]
    news_items = result["news_items"]
    financial_reports = result["financial_reports"]
    agent_results = result["agent_results"]
    report = result["report"]
    decision_summary = result.get("decision_summary") or build_decision_summary(
        portfolio,
        market_df,
        news_items,
        financial_reports=financial_reports,
        quote_metrics=result.get("quote_metrics", {}),
        prefer_llm=bool(result.get("use_llm_analysis", should_use_llm_analysis(expert_config))),
    )

    st.success(
        f"报告生成完成：行情 {len(market_df)} 条，新闻 {len(news_items)} 条，财报/公告 {len(financial_reports)} 条。"
    )

    tab_overview, tab_charts, tab_agents, tab_expert, tab_news, tab_report = st.tabs(
        ["概览", "技术图表", "智能体分析", "专家意见", "新闻与公告", "完整报告"]
    )

    with tab_overview:
        st.html(build_decision_summary_html(decision_summary))
        st.subheader("组合识别结果")
        positions_df = pd.DataFrame(portfolio["positions"])
        total_value = float((positions_df["quantity"] * positions_df["market_price"]).sum()) if not positions_df.empty else 0.0
        max_weight = (
            float(((positions_df["quantity"] * positions_df["market_price"]) / total_value).max() * 100)
            if total_value
            else 0.0
        )
        sector_count = int(positions_df["sector"].nunique()) if "sector" in positions_df else 0
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("持仓标的", len(portfolio["positions"]))
        p2.metric("组合市值", f"{total_value:,.2f}")
        p3.metric("最大单票权重", f"{max_weight:.2f}%")
        p4.metric("覆盖行业", sector_count)
        st.subheader("持仓信息")
        if not positions_df.empty:
            display_df = positions_df.copy()
            display_df["市值"] = display_df["quantity"] * display_df["market_price"]
            display_df["权重"] = display_df["市值"] / total_value * 100 if total_value else 0.0
            display_df["收益率"] = (
                (display_df["market_price"] / display_df["cost_price"] - 1) * 100
            ).where(display_df["cost_price"] > 0, 0.0)
            st.dataframe(
                display_df[
                    ["name", "symbol", "sector", "quantity", "cost_price", "market_price", "市值", "权重", "收益率"]
                ],
                column_config={
                    "name": "股票名称",
                    "symbol": "股票代码",
                    "sector": "行业",
                    "quantity": st.column_config.NumberColumn("持仓数量", format="%.0f"),
                    "cost_price": st.column_config.NumberColumn("成本价", format="%.2f"),
                    "market_price": st.column_config.NumberColumn("当前价", format="%.2f"),
                    "市值": st.column_config.NumberColumn("市值", format="%.2f"),
                    "权重": st.column_config.NumberColumn("权重(%)", format="%.2f"),
                    "收益率": st.column_config.NumberColumn("收益率(%)", format="%.2f"),
                },
                hide_index=True,
                width="stretch",
            )
        if not market_df.empty:
            source = market_df.get("data_source", pd.Series(["unknown"])).iloc[-1]
            st.write(f"行情数据源：`{format_data_source_label(str(source))}`")
            grouped_dates = market_df.groupby("symbol")["date"].agg(["min", "max"])
            c1, c2, c3 = st.columns(3)
            c1.metric("行情条数", len(market_df))
            c2.metric("行情标的", market_df["symbol"].nunique())
            c3.metric("区间覆盖", f"{grouped_dates['min'].min()} 至 {grouped_dates['max'].max()}")
        st.write(f"报告文件：`{result['report_path']}`")

    with tab_charts:
        render_technical_dashboard(result, expert_config)

    with tab_agents:
        position_by_label = {position_label(position): position for position in portfolio.get("positions", [])}
        agent_view = st.radio(
            "分析范围",
            ["组合总览", *position_by_label.keys()],
            horizontal=True,
            key="agent_scope_selector",
        )
        if agent_view == "组合总览":
            for agent_result in agent_results:
                with st.expander(agent_result.name, expanded=True):
                    if "新闻解读" in agent_result.name:
                        st.html(
                            build_agent_analysis_html(
                                agent_result.content,
                                news_items=related_news_for_portfolio(news_items, portfolio),
                            )
                        )
                    elif "财报公告" in agent_result.name:
                        st.html(
                            build_agent_analysis_html(
                                agent_result.content,
                                financial_reports=financial_reports,
                            )
                        )
                    else:
                        st.html(build_agent_analysis_html(agent_result.content))
        else:
            selected_position = position_by_label[agent_view]
            selected_symbol = str(selected_position.get("symbol"))
            news_agent = find_agent_result(agent_results, "新闻解读")
            technical_agent = find_agent_result(agent_results, "技术观察")
            risk_agent = find_agent_result(agent_results, "风险提示")
            report_agent = find_agent_result(agent_results, "财报公告")
            if news_agent:
                with st.expander("新闻解读智能体", expanded=True):
                    st.html(
                        build_agent_analysis_html(
                            news_agent.content,
                            news_items=filter_items_for_symbol(news_items, selected_symbol),
                        )
                    )
            if technical_agent:
                with st.expander("技术观察智能体", expanded=True):
                    st.html(build_agent_analysis_html(technical_agent_section_for_label(technical_agent.content, agent_view)))
            if risk_agent:
                with st.expander("风险提示智能体", expanded=True):
                    st.html(build_agent_analysis_html(risk_agent.content))
            if report_agent:
                selected_reports = filter_items_for_symbol(financial_reports, selected_symbol)
                with st.expander("财报公告智能体", expanded=False):
                    st.html(build_agent_analysis_html(report_agent.content, financial_reports=selected_reports))

    with tab_expert:
        render_expert_chat(result, expert_config)

    with tab_news:
        news_scope_options = ["全部", *[position_label(position) for position in portfolio.get("positions", [])]]
        news_scope = st.radio("信息范围", news_scope_options, horizontal=True, key="news_scope_selector")
        if news_scope == "全部":
            scoped_news = news_items
            scoped_reports = financial_reports
        else:
            selected_symbol = next(
                str(position.get("symbol"))
                for position in portfolio.get("positions", [])
                if position_label(position) == news_scope
            )
            scoped_news = filter_items_for_symbol(news_items, selected_symbol)
            scoped_reports = filter_items_for_symbol(financial_reports, selected_symbol)
        col_news, col_reports = st.columns(2)
        with col_news:
            st.subheader(f"相关新闻（{len(scoped_news)} 条）")
            st.html(build_news_cards_html(scoped_news, limit=50))
        with col_reports:
            st.subheader(f"财报/公告（{len(scoped_reports)} 条）")
            st.html(build_announcement_cards_html(scoped_reports))

    with tab_report:
        st.download_button(
            label="下载 Markdown 报告",
            data=report,
            file_name=f"{result.get('output_prefix', portfolio_output_prefix(portfolio))}_report.md",
            mime="text/markdown",
            width="stretch",
        )
        st.markdown(report)


if __name__ == "__main__":
    main()
