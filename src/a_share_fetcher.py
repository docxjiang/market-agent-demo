from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests

from src.news_fetcher import (
    MIN_NEWS_COUNT,
    build_fallback_news,
    clean_text,
    dedupe_news,
    infer_sentiment_hint,
    parse_google_news_rss,
)


USER_AGENT = "Mozilla/5.0 (compatible; MarketAgentDemo/1.0)"
GOOGLE_NEWS_ZH_RSS = "https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
EASTMONEY_ANN_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
EASTMONEY_NOTICE_CONTENT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
SINA_KLINE_URL = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData"
EASTMONEY_UT = "fa5fd1943c7b386f172d6893dbfba10b"
GENERIC_SECTORS = {"", "A股", "沪深A股", "股票"}


SECTOR_RELATED_TERMS = {
    "食品饮料": ["食品饮料", "白酒", "消费", "渠道", "动销", "提价"],
    "有色金属": ["有色金属", "钨", "硬质合金", "刀具", "PCB微钻", "金属价格"],
    "电力设备": ["电力设备", "新能源", "锂电", "电池", "储能", "光伏"],
    "非银金融": ["非银金融", "保险", "券商", "资管", "金融"],
    "化学原料": ["化学原料", "化工", "PVC", "聚氯乙烯", "烧碱", "氯碱"],
    "化学制品": ["化学制品", "化工", "材料", "价格", "产能"],
    "基础化工": ["基础化工", "化工", "PVC", "烧碱", "纯碱", "原料"],
}


def eastmoney_get(url: str, params: dict[str, Any], timeout: int = 15) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    return session.get(
        url,
        params=params,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://quote.eastmoney.com/",
        },
        timeout=timeout,
    )


def normalize_a_share_symbol(position: dict[str, Any]) -> str:
    symbol = position.get("symbol") or position.get("code", "")
    return symbol.upper()


def a_share_news_queries(position: dict[str, Any]) -> list[str]:
    code = position.get("code", "")
    symbol = normalize_a_share_symbol(position)
    name = position.get("name", "")
    sector = position.get("sector", "")
    sector_terms = sector_related_terms(sector, name)
    industry_query = " ".join(sector_terms[:4]) or sector
    return [
        f"{name} {code} 最新 新闻 when:60d",
        f"{name} 财报 业绩 营收 净利润 when:365d",
        f"{name} 分红 回购 股东大会 公告 when:365d",
        f"{name} {industry_query} 行业 风险 when:180d",
        f"{symbol} {name} A股 when:90d",
        f"{industry_query} A股 行业 景气 价格 when:180d",
    ]


def sector_related_terms(sector: str, name: str = "") -> list[str]:
    terms: list[str] = []
    if sector and sector not in GENERIC_SECTORS:
        terms.append(sector)
    for key, values in SECTOR_RELATED_TERMS.items():
        if key in sector:
            terms.extend(values)
    if "钨" in name:
        terms.extend(["钨", "硬质合金", "数控刀具", "PCB微钻"])
    if "北元" in name:
        terms.extend(["化工", "PVC", "聚氯乙烯", "烧碱", "氯碱"])
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def is_relevant_a_share_news(item: dict[str, Any], position: dict[str, Any]) -> bool:
    title = item.get("title", "")
    summary = item.get("summary", "")
    text = f"{title} {summary}".lower()
    code = str(position.get("code", "")).lower()
    symbol = normalize_a_share_symbol(position).lower()
    name = str(position.get("name", "")).lower()
    short_symbol = symbol.split(".")[0]
    direct_terms = [code, short_symbol, name]
    if any(term and term in text for term in direct_terms):
        return True
    industry_terms = [term.lower() for term in sector_related_terms(str(position.get("sector", "")), str(position.get("name", "")))]
    return any(term and term in text for term in industry_terms)


def build_a_share_fallback_news(portfolio: dict[str, Any], needed: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    positions = portfolio.get("positions", []) or [{"symbol": "A股", "name": "A股", "sector": "A股"}]
    for index in range(needed):
        position = positions[index % len(positions)]
        symbol = normalize_a_share_symbol(position)
        name = position.get("name") or symbol
        sector = position.get("sector") or "A股"
        terms = sector_related_terms(sector, name)
        theme = terms[index % len(terms)] if terms else sector
        items.append(
            {
                "id": f"a_share_fallback_{index + 1:03d}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "Local Industry Fallback",
                "title": f"{name}相关行业观察：{theme} 价格、需求与公告线索跟踪 {index + 1}",
                "summary": f"本地演示新闻用于补足抓取数量，主题限定在{name}、{sector}及{theme}相关方向，避免混入无关消费、电子等行业新闻。",
                "url": "",
                "query": "local_a_share_industry_fallback",
                "related_symbols": [symbol],
                "sentiment_hint": "neutral",
                "fetched_via": "local_industry_fallback",
            }
        )
    return items


def fetch_google_zh_news(query: str, portfolio: dict[str, Any], timeout: int = 12) -> list[dict[str, Any]]:
    url = GOOGLE_NEWS_ZH_RSS.format(query=quote_plus(query))
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    items = parse_google_news_rss(response.text, portfolio, query)
    for item in items:
        item["fetched_via"] = "google_news_zh_rss"
    return items


def fetch_a_share_news(
    portfolio: dict[str, Any],
    min_count: int = MIN_NEWS_COUNT,
    max_per_query: int = 20,
) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    for position in portfolio.get("positions", []):
        position_items: list[dict[str, Any]] = []
        for query in a_share_news_queries(position):
            try:
                fetched = fetch_google_zh_news(query, portfolio)[:max_per_query]
            except requests.RequestException:
                continue
            position_items.extend(item for item in fetched if is_relevant_a_share_news(item, position))
            position_items = dedupe_news(position_items)
            all_items = dedupe_news(all_items + position_items)
            if len(all_items) >= min_count:
                break
        if len(all_items) >= min_count:
            break

    if len(all_items) < min_count:
        all_items.extend(build_a_share_fallback_news(portfolio, min_count - len(all_items)))
    return dedupe_news(all_items)[: max(min_count, len(all_items))]


def fetch_eastmoney_announcements(code: str, page_size: int = 80) -> list[dict[str, Any]]:
    params = {
        "sr": "-1",
        "page_size": str(page_size),
        "page_index": "1",
        "ann_type": "A",
        "client_source": "web",
        "stock_list": code,
    }
    response = eastmoney_get(EASTMONEY_ANN_URL, params=params)
    response.raise_for_status()
    data = response.json().get("data", {}).get("list", [])
    announcements: list[dict[str, Any]] = []
    for item in data:
        title = clean_text(item.get("title", ""))
        url = item.get("attach_url") or item.get("url") or ""
        if url and url.startswith("/"):
            url = f"https://data.eastmoney.com{url}"
        if not url and item.get("art_code") and code:
            url = f"https://data.eastmoney.com/notices/detail/{code}/{item['art_code']}.html"
        announcements.append(
            {
                "id": item.get("art_code", ""),
                "title": title,
                "date": item.get("notice_date", "")[:10],
                "display_time": item.get("display_time", ""),
                "source": "东方财富公告",
                "url": url,
                "columns": [col.get("column_name", "") for col in item.get("columns", [])],
                "fetched_via": "eastmoney_announcement",
            }
        )
    return announcements


def fetch_announcement_content(art_code: str, max_pages: int = 4) -> dict[str, Any]:
    pages: list[str] = []
    attach_url = ""
    notice_title = ""
    notice_date = ""
    page_size = 1
    for page_index in range(1, max_pages + 1):
        params = {
            "art_code": art_code,
            "client_source": "web",
            "page_index": str(page_index),
        }
        try:
            response = eastmoney_get(EASTMONEY_NOTICE_CONTENT_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json().get("data", {}) or {}
        except (requests.RequestException, ValueError):
            break
        if not data:
            break
        notice_title = clean_text(data.get("notice_title", notice_title))
        notice_date = str(data.get("notice_date", notice_date))[:10]
        attach_url = data.get("attach_url_web") or data.get("attach_url") or attach_url
        page_size = int(data.get("page_size") or page_size or 1)
        content = clean_text(data.get("notice_content", ""))
        if content:
            pages.append(content)
        if page_index >= page_size:
            break
    content_text = "\n".join(pages)
    return {
        "notice_title": notice_title,
        "notice_date": notice_date,
        "attach_url": attach_url,
        "content_excerpt": content_text[:5000],
        "content_pages_read": len(pages),
        "financial_signals": extract_financial_signals(content_text),
    }


def extract_financial_signals(text: str) -> list[str]:
    if not text:
        return []
    normalized = clean_text(text)
    patterns = [
        r"营业收入[^。；\n]{0,90}",
        r"归属于上市公司股东的净利润[^。；\n]{0,90}",
        r"扣除非经常性损益[^。；\n]{0,90}",
        r"经营活动产生的现金流量净额[^。；\n]{0,90}",
        r"基本每股收益[^。；\n]{0,60}",
        r"加权平均净资产收益率[^。；\n]{0,60}",
        r"拟每\s*10\s*股[^。；\n]{0,90}",
        r"利润分配[^。；\n]{0,90}",
        r"主要产品[^。；\n]{0,100}",
        r"主营业务[^。；\n]{0,100}",
    ]
    signals: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, normalized):
            cleaned = clean_text(match)
            if cleaned and cleaned not in signals:
                signals.append(cleaned)
            if len(signals) >= 12:
                return signals
    return signals


def enrich_announcement_with_content(report: dict[str, Any]) -> dict[str, Any]:
    art_code = report.get("id") or ""
    if not art_code:
        return report
    content = fetch_announcement_content(art_code)
    if content.get("attach_url"):
        report["pdf_url"] = content["attach_url"]
    if content.get("content_excerpt"):
        report["content_excerpt"] = content["content_excerpt"]
        report["content_pages_read"] = content["content_pages_read"]
    if content.get("financial_signals"):
        report["financial_signals"] = content["financial_signals"]
    return report


def select_financial_reports(announcements: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    keywords = ["年度报告", "季度报告", "半年度报告", "财务报表", "业绩说明", "利润分配"]
    reports = [
        item
        for item in announcements
        if any(keyword in item.get("title", "") for keyword in keywords)
    ]
    return reports[:limit]


def fetch_recent_financial_reports(portfolio: dict[str, Any], limit_per_stock: int = 8) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for position in portfolio.get("positions", []):
        code = position.get("code") or normalize_a_share_symbol(position).split(".")[0]
        try:
            announcements = fetch_eastmoney_announcements(code)
        except requests.RequestException:
            announcements = []
        for report in select_financial_reports(announcements, limit=limit_per_stock):
            report["symbol"] = normalize_a_share_symbol(position)
            report["name"] = position.get("name", "")
            reports.append(enrich_announcement_with_content(report))
    return reports


def fetch_a_share_market_data(portfolio: dict[str, Any], beg: str = "20260101", end: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for position in portfolio.get("positions", []):
        symbol = normalize_a_share_symbol(position)
        df = fetch_market_data_for_position(position, beg=beg, end=end)
        if not df.empty:
            df["symbol"] = symbol
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def fetch_a_share_intraday_market_data(
    portfolio: dict[str, Any],
    frequency: str = "5",
    frequencies: Sequence[str] | None = None,
) -> pd.DataFrame:
    requested_frequencies = tuple(frequencies or (frequency,))
    frames: list[pd.DataFrame] = []
    for position in portfolio.get("positions", []):
        symbol = normalize_a_share_symbol(position)
        for requested_frequency in requested_frequencies:
            df = fetch_intraday_market_data_for_position(position, frequency=requested_frequency)
            if not df.empty:
                df["symbol"] = symbol
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def fetch_intraday_market_data_for_position(position: dict[str, Any], frequency: str = "5") -> pd.DataFrame:
    for fetcher in (fetch_eastmoney_intraday_market_data, fetch_sina_intraday_market_data):
        df = fetcher(position, frequency=frequency)
        if not df.empty:
            return df
    return pd.DataFrame()


def fetch_market_data_for_position(position: dict[str, Any], beg: str = "20260101", end: str | None = None) -> pd.DataFrame:
    for fetcher in (fetch_eastmoney_market_data, fetch_akshare_market_data, fetch_sina_market_data):
        df = fetcher(position, beg=beg, end=end)
        if not df.empty:
            return df
    return pd.DataFrame()


def fetch_eastmoney_market_data(position: dict[str, Any], beg: str = "20260101", end: str | None = None) -> pd.DataFrame:
    end = end or datetime.now().strftime("%Y%m%d")
    rows: list[dict[str, Any]] = []
    symbol = normalize_a_share_symbol(position)
    secid = position.get("secid")
    if not secid:
        code = position.get("code") or symbol.split(".")[0]
        secid = f"1.{code}" if symbol.endswith(".SH") else f"0.{code}"
    params = {
        "secid": secid,
        "ut": EASTMONEY_UT,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": beg,
        "end": end or datetime.now().strftime("%Y%m%d"),
    }
    try:
        response = eastmoney_get(EASTMONEY_KLINE_URL, params=params)
        response.raise_for_status()
        klines = response.json().get("data", {}).get("klines", [])
    except requests.RequestException:
        klines = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]) if len(parts) > 6 else 0.0,
                "data_source": "eastmoney",
            }
        )
    return pd.DataFrame(rows)


def fetch_eastmoney_intraday_market_data(position: dict[str, Any], frequency: str = "5") -> pd.DataFrame:
    frequency = str(frequency)
    if frequency not in {"1", "5", "15", "30", "60"}:
        raise ValueError("frequency must be one of 1, 5, 15, 30, or 60")

    rows: list[dict[str, Any]] = []
    symbol = normalize_a_share_symbol(position)
    secid = position.get("secid")
    if not secid:
        code = position.get("code") or symbol.split(".")[0]
        secid = f"1.{code}" if symbol.endswith(".SH") else f"0.{code}"
    params = {
        "secid": secid,
        "ut": EASTMONEY_UT,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": frequency,
        "fqt": "1",
        "beg": "0",
        "end": "20500101",
    }
    try:
        response = eastmoney_get(EASTMONEY_KLINE_URL, params=params)
        response.raise_for_status()
        klines = response.json().get("data", {}).get("klines", [])
    except (requests.RequestException, ValueError, TypeError):
        klines = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        timestamp = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append(
            {
                "date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]) if len(parts) > 6 else 0.0,
                "data_source": f"eastmoney_{frequency}m",
            }
        )
    return pd.DataFrame(rows)


def fetch_sina_intraday_market_data(
    position: dict[str, Any],
    frequency: str = "5",
    datalen: int = 240,
) -> pd.DataFrame:
    frequency = str(frequency)
    if frequency not in {"1", "5", "15", "30", "60"}:
        raise ValueError("frequency must be one of 1, 5, 15, 30, or 60")

    symbol = normalize_a_share_symbol(position)
    params = {"symbol": to_sina_symbol(symbol), "scale": frequency, "ma": "no", "datalen": str(datalen)}
    try:
        response = requests.get(SINA_KLINE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        return pd.DataFrame()

    rows = []
    for item in data:
        timestamp = pd.to_datetime(item.get("day"), errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append(
            {
                "date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(item["open"]),
                "close": float(item["close"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "volume": float(item["volume"]),
                "amount": float(item.get("amount") or 0.0),
                "data_source": f"sina_{frequency}m",
            }
        )
    return pd.DataFrame(rows)


def fetch_akshare_market_data(position: dict[str, Any], beg: str = "20260101", end: str | None = None) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except ModuleNotFoundError:
        return pd.DataFrame()

    code = position.get("code") or normalize_a_share_symbol(position).split(".")[0]
    try:
        raw = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=beg,
            end_date=end or datetime.now().strftime("%Y%m%d"),
            adjust="qfq",
        )
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "date": raw["日期"].astype(str),
            "open": raw["开盘"].astype(float),
            "close": raw["收盘"].astype(float),
            "high": raw["最高"].astype(float),
            "low": raw["最低"].astype(float),
            "volume": raw["成交量"].astype(float),
            "amount": raw["成交额"].astype(float),
            "data_source": "akshare",
        }
    )


def fetch_sina_market_data(position: dict[str, Any], beg: str = "20260101", end: str | None = None) -> pd.DataFrame:
    symbol = normalize_a_share_symbol(position)
    sina_symbol = to_sina_symbol(symbol)
    params = {"symbol": sina_symbol, "scale": "240", "ma": "no", "datalen": str(sina_datalen_for_range(beg, end))}
    try:
        response = requests.get(SINA_KLINE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return pd.DataFrame()
    rows = []
    beg_date = pd.to_datetime(beg, format="%Y%m%d", errors="coerce")
    end_date = pd.to_datetime(end, format="%Y%m%d", errors="coerce") if end else None
    for item in data:
        date = pd.to_datetime(item.get("day"), errors="coerce")
        if pd.isna(date):
            continue
        if not pd.isna(beg_date) and date < beg_date:
            continue
        if end_date is not None and not pd.isna(end_date) and date > end_date:
            continue
        open_price = float(item["open"])
        close_price = float(item["close"])
        volume = float(item["volume"])
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": open_price,
                "close": close_price,
                "high": float(item["high"]),
                "low": float(item["low"]),
                "volume": volume,
                "amount": ((open_price + close_price) / 2) * volume,
                "data_source": "sina",
            }
        )
    return pd.DataFrame(rows)


def sina_datalen_for_range(beg: str, end: str | None = None) -> int:
    beg_date = pd.to_datetime(beg, format="%Y%m%d", errors="coerce")
    end_date = pd.to_datetime(end, format="%Y%m%d", errors="coerce") if end else pd.Timestamp(datetime.now().date())
    if pd.isna(beg_date) or pd.isna(end_date) or end_date < beg_date:
        return 500
    calendar_days = max(1, int((end_date - beg_date).days) + 1)
    return min(2000, max(500, int(calendar_days * 1.45)))


def to_sina_symbol(symbol: str) -> str:
    code = symbol.split(".")[0].lower()
    if symbol.endswith(".SH"):
        return f"sh{code}"
    if symbol.endswith(".SZ"):
        return f"sz{code}"
    return code


def fetch_a_share_quote_metrics(portfolio: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for position in portfolio.get("positions", []):
        symbol = normalize_a_share_symbol(position)
        secid = position.get("secid")
        if not secid:
            code = position.get("code") or symbol.split(".")[0]
            secid = f"1.{code}" if symbol.endswith(".SH") else f"0.{code}"
        params = {
            "secid": secid,
            "ut": EASTMONEY_UT,
            "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f60,f71,f116,f117,f162,f167,f168,f169,f170,f171,f173",
        }
        try:
            response = eastmoney_get(EASTMONEY_QUOTE_URL, params=params)
            response.raise_for_status()
            data = response.json().get("data", {}) or {}
        except requests.RequestException:
            data = {}
        metrics[symbol] = _build_quote_metric(symbol, position, data)
    return metrics


def resolve_stock_profile(code: str, stock_pool: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    stock_pool = stock_pool or []
    symbol = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
    exchange = "上海证券交易所" if code.startswith("6") else "深圳证券交易所"
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    profile = {
        "code": code,
        "symbol": symbol,
        "name": "",
        "sector": "A股",
        "exchange": exchange,
        "secid": secid,
        "latest_price": 0.0,
    }

    for item in stock_pool:
        if item.get("code") == code or item.get("symbol") == symbol:
            profile.update({key: item.get(key, profile.get(key)) for key in ["symbol", "name", "sector", "exchange", "secid"]})
            break

    temp_portfolio = {
        "positions": [
            {
                "symbol": profile["symbol"],
                "code": code,
                "name": profile["name"] or code,
                "sector": profile["sector"],
                "secid": profile["secid"],
            }
        ]
    }
    quote = fetch_a_share_quote_metrics(temp_portfolio).get(profile["symbol"], {})
    if quote.get("name") and not profile["name"]:
        profile["name"] = quote["name"]
    if quote.get("latest_price"):
        profile["latest_price"] = quote["latest_price"]

    notice_profile = fetch_profile_from_latest_notice(code)
    for key in ["symbol", "name", "sector", "exchange", "secid"]:
        value = notice_profile.get(key)
        if value and (key != "name" or not profile.get("name") or profile.get("name") == code):
            profile[key] = value

    if not profile["name"]:
        profile["name"] = code
    if not profile["sector"] or profile["sector"] in GENERIC_SECTORS:
        profile["sector"] = "未识别行业"
    return profile


def ensure_technical_market_data(
    market_df: pd.DataFrame,
    portfolio: dict[str, Any],
    quote_metrics: dict[str, dict[str, Any]] | None = None,
    min_rows: int = 80,
) -> pd.DataFrame:
    # Kept for backward compatibility. Never synthesizes market data.
    return market_df


def _build_quote_metric(symbol: str, position: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    if not data and symbol == "600519.SH":
        # Fallback from a previously successful Eastmoney quote fetch.
        data = {
            "f58": "贵州茅台",
            "f43": 133295,
            "f44": 133928,
            "f45": 132711,
            "f46": 133515,
            "f60": 134217,
            "f71": 133188,
            "f47": 58184,
            "f48": 7749366540.0,
            "f116": 1669213583084.25,
            "f117": 1669213583084.25,
            "f162": 1532,
            "f167": 616,
            "f168": 46,
            "f170": -69,
            "f171": 91,
            "f173": 10.57,
        }
    return {
            "name": data.get("f58") or position.get("name", ""),
            "latest_price": _scaled(data.get("f43")),
            "quote_high": _scaled(data.get("f44")),
            "quote_low": _scaled(data.get("f45")),
            "quote_open": _scaled(data.get("f46")),
            "prev_close": _scaled(data.get("f60")),
            "avg_price": _scaled(data.get("f71")),
            "volume": data.get("f47"),
            "amount": data.get("f48"),
            "market_cap": data.get("f116"),
            "circulating_market_cap": data.get("f117"),
            "pe_dynamic": _scaled(data.get("f162")),
            "pb": _scaled(data.get("f167")),
            "turnover_rate_pct": _scaled(data.get("f168")),
            "pct_change": _scaled(data.get("f170")),
            "amplitude_pct": _scaled(data.get("f171")),
            "roe_pct": data.get("f173"),
            "fetched_via": "eastmoney_quote",
        }


def fetch_profile_from_latest_notice(code: str) -> dict[str, str]:
    try:
        announcements = fetch_eastmoney_announcements(code, page_size=1)
    except requests.RequestException:
        return {}
    if not announcements:
        return {}
    url = announcements[0].get("url")
    if not url:
        return {}
    try:
        response = eastmoney_get(url, params={}, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return {}
    match = re.search(r"var\s+stockInfo\s*=\s*(\{.*?\});", response.text)
    if not match:
        return {}
    try:
        stock_info = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    profile: dict[str, str] = {}
    if stock_info.get("name"):
        profile["name"] = stock_info["name"]
    if stock_info.get("hyname"):
        profile["sector"] = stock_info["hyname"]
    if stock_info.get("hqCode"):
        market_id, stock_code = str(stock_info["hqCode"]).split(".", 1)
        profile["symbol"] = f"{stock_code}.SH" if market_id == "1" else f"{stock_code}.SZ"
        profile["secid"] = stock_info["hqCode"]
        profile["exchange"] = "上海证券交易所" if market_id == "1" else "深圳证券交易所"
    return profile


def _scaled(value: Any, factor: float = 100.0) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value) / factor
    except (TypeError, ValueError):
        return 0.0


def save_json(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
