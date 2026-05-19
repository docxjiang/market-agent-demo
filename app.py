from __future__ import annotations

import json
from datetime import date
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.agents import run_all_agents
from src.a_share_fetcher import (
    fetch_a_share_market_data,
    fetch_a_share_news,
    fetch_a_share_quote_metrics,
    fetch_recent_financial_reports,
    resolve_stock_profile,
    save_json,
)
from src.charting import generate_technical_charts
from src.expert import llm_expert_reply, local_expert_reply
from src.llm_client import LLMConfig
from src.news_fetcher import MIN_NEWS_COUNT
from src.report import build_full_news_report


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DISCLAIMER = "免责声明：本系统仅用于课程演示和市场信息解读，不构成投资建议。"


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


def build_portfolio(profile: dict[str, Any], quantity: float, cost_price: float) -> dict[str, Any]:
    market_price = float(profile.get("latest_price") or cost_price)
    symbol = profile["symbol"]
    return {
        "user_id": "streamlit_user",
        "base_currency": "CNY",
        "as_of_date": date.today().isoformat(),
        "risk_profile": "moderate",
        "market": "A股",
        "positions": [
            {
                "symbol": symbol,
                "code": profile["code"],
                "name": profile["name"],
                "asset_type": "A股",
                "quantity": quantity,
                "cost_price": cost_price,
                "market_price": market_price,
                "sector": profile["sector"],
                "exchange": profile["exchange"],
                "secid": profile["secid"],
            }
        ],
    }


def output_prefix(code: str, name: str) -> str:
    safe_name = "".join(ch for ch in name if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return f"{code}_{safe_name or 'stock'}"


@st.cache_data(show_spinner=False, ttl=600)
def generate_report_cached(
    code: str,
    quantity: float,
    cost_price: float,
) -> dict[str, Any]:
    profile = resolve_stock_profile(code, load_stock_pool())
    portfolio = build_portfolio(profile, quantity, cost_price)
    symbol = portfolio["positions"][0]["symbol"]
    prefix = output_prefix(code, profile["name"])

    market_df = fetch_a_share_market_data(portfolio, beg="20250101")
    quote_metrics = fetch_a_share_quote_metrics(portfolio)
    if not market_df.empty and not profile.get("latest_price"):
        latest_close = float(market_df.sort_values("date").iloc[-1]["close"])
        profile["latest_price"] = latest_close
        portfolio["positions"][0]["market_price"] = latest_close
    news_items = fetch_a_share_news(portfolio, min_count=MIN_NEWS_COUNT)
    financial_reports = fetch_recent_financial_reports(portfolio)

    chart_paths: dict[str, dict[str, str]] = {}
    chart_abs_paths: dict[str, dict[str, str]] = {}
    if not market_df.empty:
        generated = generate_technical_charts(market_df, symbol, OUTPUT_DIR / "charts")
        chart_paths[symbol] = {
            key: str(path.relative_to(BASE_DIR)).replace("\\", "/")
            for key, path in generated.items()
        }
        chart_abs_paths[symbol] = {key: str(path) for key, path in generated.items()}

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(news_items, OUTPUT_DIR / f"{prefix}_news.json")
    save_json(financial_reports, OUTPUT_DIR / f"{prefix}_financial_reports.json")
    (OUTPUT_DIR / f"{prefix}_quote_metrics.json").write_text(
        json.dumps(quote_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not market_df.empty:
        market_df.to_csv(OUTPUT_DIR / f"{prefix}_market_for_indicators.csv", index=False, encoding="utf-8")
    report_path = OUTPUT_DIR / f"{prefix}_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "portfolio": portfolio,
        "profile": profile,
        "market_df": market_df,
        "quote_metrics": quote_metrics,
        "news_items": news_items,
        "financial_reports": financial_reports,
        "agent_results": agent_results,
        "report": report,
        "report_path": str(report_path),
        "chart_abs_paths": chart_abs_paths,
    }


def sidebar_inputs() -> dict[str, Any]:
    st.sidebar.header("股票与持仓")
    preset = st.sidebar.selectbox(
        "快速示例",
        ["中钨高新 000657", "贵州茅台 600519", "自定义"],
        index=0,
    )
    defaults = {
        "中钨高新 000657": ("000657", 1000.0, 50.0),
        "贵州茅台 600519": ("600519", 100.0, 1500.0),
        "自定义": ("000657", 1000.0, 10.0),
    }
    code, quantity, cost_price = defaults[preset]

    code = st.sidebar.text_input("股票代码", value=code, max_chars=6)
    quantity = st.sidebar.number_input("持仓数量", value=float(quantity), min_value=0.0, step=100.0)
    cost_price = st.sidebar.number_input("成本价", value=float(cost_price), min_value=0.0, step=0.01)
    st.sidebar.caption("只需输入 6 位 A 股代码。股票名称、行业、当前价、交易所会自动解析。")

    return {
        "code": code.strip(),
        "quantity": float(quantity),
        "cost_price": float(cost_price),
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


def render_charts(chart_abs_paths: dict[str, dict[str, str]]) -> None:
    if not chart_abs_paths:
        st.warning("未生成技术图表：行情数据不足或数据源暂不可用。")
        return
    for symbol, paths in chart_abs_paths.items():
        st.subheader(f"{symbol} 技术图表")
        cols = st.columns(1)
        if paths.get("kline_ma"):
            cols[0].image(paths["kline_ma"], caption="K 线叠加 MA5 / MA10 / MA20")
        col_macd, col_amount = st.columns(2)
        if paths.get("macd"):
            col_macd.image(paths["macd"], caption="MACD（12/26/9）")
        if paths.get("amount"):
            col_amount.image(paths["amount"], caption="成交额")


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
        if st.button("清空专家对话", use_container_width=True):
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
    )

    st.title("A股个人持仓新闻解读与风险提醒系统")
    st.caption("输入任意 A 股标的，自动抓取新闻、财报公告、行情数据，生成多智能体分析报告。")

    inputs = sidebar_inputs()
    expert_config = sidebar_expert_config()
    code = inputs["code"]

    if not code or len(code) != 6 or not code.isdigit():
        st.error("请在左侧输入 6 位 A 股代码。")
        return

    generate = st.sidebar.button("生成分析报告", type="primary", use_container_width=True)
    clear_cache = st.sidebar.button("清除缓存后重新抓取", use_container_width=True)
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
        with st.spinner("正在抓取行情、新闻、财报公告并运行多智能体分析..."):
            st.session_state.latest_result = generate_report_cached(**inputs)

    result = st.session_state.latest_result
    portfolio = result["portfolio"]
    profile = result["profile"]
    market_df: pd.DataFrame = result["market_df"]
    news_items = result["news_items"]
    financial_reports = result["financial_reports"]
    agent_results = result["agent_results"]
    report = result["report"]

    st.success(
        f"报告生成完成：行情 {len(market_df)} 条，新闻 {len(news_items)} 条，财报/公告 {len(financial_reports)} 条。"
    )

    tab_overview, tab_charts, tab_agents, tab_expert, tab_news, tab_report = st.tabs(
        ["概览", "技术图表", "智能体分析", "专家意见", "新闻与公告", "完整报告"]
    )

    with tab_overview:
        st.subheader("自动识别结果")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("股票名称", profile.get("name", "未知"))
        p2.metric("股票代码", profile.get("symbol", code))
        p3.metric("所属行业", profile.get("sector", "未识别行业"))
        latest_price = profile.get("latest_price") or portfolio["positions"][0]["market_price"]
        p4.metric("自动获取当前价", f"{float(latest_price):.2f}")
        st.subheader("持仓信息")
        st.dataframe(pd.DataFrame(portfolio["positions"]), use_container_width=True)
        if not market_df.empty:
            source = market_df.get("data_source", pd.Series(["unknown"])).iloc[-1]
            st.write(f"行情数据源：`{source}`")
            first = market_df.iloc[0]
            last = market_df.iloc[-1]
            change_pct = (float(last["close"]) / float(first["close"]) - 1) * 100
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("行情条数", len(market_df))
            c2.metric("行情区间", f"{first['date']} 至 {last['date']}")
            c3.metric("起始收盘价", f"{float(first['close']):.2f}")
            c4.metric("最新收盘价", f"{float(last['close']):.2f}")
            c5.metric("区间涨跌幅", f"{change_pct:.2f}%")
        st.write(f"报告文件：`{result['report_path']}`")

    with tab_charts:
        render_charts(result["chart_abs_paths"])

    with tab_agents:
        for agent_result in agent_results:
            with st.expander(agent_result.name, expanded=True):
                st.markdown(agent_result.content)

    with tab_expert:
        render_expert_chat(result, expert_config)

    with tab_news:
        col_news, col_reports = st.columns(2)
        with col_news:
            st.subheader(f"相关新闻（{len(news_items)} 条）")
            for item in news_items[:30]:
                st.markdown(f"**{item.get('title', '未命名新闻')}**")
                st.caption(f"{item.get('source', '未知')} | {item.get('date', '未知')} | {item.get('sentiment_hint', 'neutral')}")
        with col_reports:
            st.subheader(f"财报/公告（{len(financial_reports)} 条）")
            for item in financial_reports:
                st.markdown(f"**{item.get('title', '未命名公告')}**")
                st.caption(f"{item.get('date', '未知')} | {item.get('source', '未知')}")

    with tab_report:
        st.download_button(
            label="下载 Markdown 报告",
            data=report,
            file_name=f"{output_prefix(code, profile.get('name', code))}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown(report)


if __name__ == "__main__":
    main()
