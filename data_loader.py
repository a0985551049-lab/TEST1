"""資料層：全部走 FinMind（美股指數例外，用 yfinance 並延遲載入）。"""
import datetime as dt
import re

import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

TTL = 3600
INST_MAP = {"Foreign_Investor": "外資", "Foreign_Dealer_Self": "外資", "Investment_Trust": "投信"}


def secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:  # 沒有 secrets.toml
        return ""


@st.cache_resource
def _dl() -> DataLoader:
    token = secret("FINMIND_TOKEN")  # 有 token 額度 600 次/時，沒有走匿名 300 次/時
    try:
        return DataLoader(token=token)
    except Exception:
        return DataLoader()


def _since(days: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def is_rate_limit(e: Exception) -> bool:
    m = str(e)
    return any(k in m for k in ("402", "429", "Payment Required", "rate limit"))


# ───────────────────────── 股票基本資料 ─────────────────────────
@st.cache_data(ttl=86400)
def stock_info(stock_id: str) -> dict:
    """回傳 {'name':..., 'industry':...}；整張表一天只抓一次。"""
    try:
        df = _dl().taiwan_stock_info()
        row = df[df["stock_id"] == stock_id]
        if not row.empty:
            r = row.iloc[0]
            return {"name": r.get("stock_name", stock_id), "industry": str(r.get("industry_category", ""))}
    except Exception:
        pass
    return {"name": stock_id, "industry": ""}


def is_financial(stock_id: str) -> bool:
    ind = stock_info(stock_id)["industry"]
    return any(k in ind for k in ("金融", "保險", "銀行", "證券")) or stock_id.startswith(("28", "58"))


# ───────────────────────── 日線＋籌碼 ─────────────────────────
@st.cache_data(ttl=TTL)
def price_data(stock_id: str, days: int, adjusted: bool = True):
    """回傳 (df, warnings)。df 欄位：date open high low close volume MA* 外資 投信 融資餘額。"""
    dl, warns = _dl(), []
    start = _since(int(days * 1.5) + 380)  # 多抓約一年，確保 MA240 在整個顯示區間都有值

    df = None
    if adjusted:
        try:
            df = dl.taiwan_stock_daily_adj(stock_id=stock_id, start_date=start)
        except Exception:
            df = None
        if df is None or df.empty:
            warns.append("adj_fallback")
            df = None
    if df is None:
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start)
    if df is None or df.empty:
        return None, ["no_data"]

    df = df.rename(columns={"Trading_Volume": "volume", "max": "high", "min": "low"})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = (df["volume"] / 1000).round().astype(int)  # 股 → 張
    df = df.sort_values("date").reset_index(drop=True)
    for p in (5, 20, 60, 100, 120, 240):
        df[f"MA{p}"] = df["close"].rolling(p).mean()

    # 三大法人
    try:
        inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not inst.empty:
            inst["net"] = (inst["buy"] - inst["sell"]) / 1000
            inst["name"] = inst["name"].map(INST_MAP)
            piv = inst.dropna(subset=["name"]).pivot_table(index="date", columns="name", values="net", aggfunc="sum")
            piv.index = pd.to_datetime(piv.index)
            df = df.merge(piv.reset_index(), on="date", how="left")
    except Exception as e:
        warns.append("rate_limit" if is_rate_limit(e) else "inst_error")

    # 融資
    try:
        mg = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start)
        if not mg.empty:
            mg = mg[["date", "MarginPurchaseTodayBalance"]].rename(columns={"MarginPurchaseTodayBalance": "融資餘額"})
            mg["date"] = pd.to_datetime(mg["date"])
            df = df.merge(mg, on="date", how="left")
    except Exception as e:
        warns.append("rate_limit" if is_rate_limit(e) else "margin_error")

    for c in ("外資", "投信"):
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).round().astype(int)
    if "融資餘額" in df:
        df["融資餘額"] = pd.to_numeric(df["融資餘額"], errors="coerce")
    return df.tail(days).reset_index(drop=True), warns


# ───────────────────────── 大戶持股比 ─────────────────────────
@st.cache_data(ttl=TTL)
def holding_ratio(stock_id: str):
    """大戶（>400張）持股比例週資料：DataFrame[date, 大戶持股比例]，失敗回傳 None。"""
    try:
        h = _dl().taiwan_stock_holding_shares_per(stock_id=stock_id, start_date=_since(400))
        if h is None or h.empty:
            return None

        def lower(s):
            n = re.findall(r"[\d,]+", str(s))
            return int(n[0].replace(",", "")) if n else -1

        h = h[h["HoldingSharesLevel"].map(lower) >= 400001]
        out = h.groupby("date", as_index=False)["percent"].sum().rename(columns={"percent": "大戶持股比例"})
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date").reset_index(drop=True) if not out.empty else None
    except Exception:
        return None


# ───────────────────────── 月營收 / 季報 ─────────────────────────
@st.cache_data(ttl=TTL)
def monthly_revenue(stock_id: str):
    """近 36 個月月營收：DataFrame[日期, 營收(元), 月增率, 年增率]。"""
    try:
        r = _dl().taiwan_stock_month_revenue(stock_id=stock_id, start_date=_since(365 * 4))
        if r is None or r.empty:
            return None
        r["日期"] = pd.to_datetime(dict(year=r["revenue_year"], month=r["revenue_month"], day=1))
        r = r.sort_values("日期")[["日期", "revenue"]].rename(columns={"revenue": "營收"})
        r["月增率"] = r["營收"].pct_change() * 100
        r["年增率"] = r["營收"].pct_change(12) * 100
        return r.tail(36).reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=TTL)
def quarterly(stock_id: str):
    """近 12 季：DataFrame[季度, 營收(元), 毛利率]。金融股以月營收加總、不算毛利率。"""
    if is_financial(stock_id):
        m = monthly_revenue(stock_id)
        if m is None:
            return None
        q = m.groupby(m["日期"].dt.to_period("Q"))["營收"].agg(["sum", "count"])
        q = q[q["count"] == 3]  # 只保留三個月都齊的季
        return pd.DataFrame({"季度": q.index.astype(str), "營收": q["sum"].values, "毛利率": float("nan")}).tail(12)

    try:
        f = _dl().taiwan_stock_financial_statement(stock_id=stock_id, start_date=_since(365 * 4))
        if f is None or f.empty:
            return None
        piv = f.pivot_table(index="date", columns="type", values="value", aggfunc="first")
        if "Revenue" not in piv:
            return None
        rev = piv["Revenue"]
        if "GrossProfit" in piv:
            gp = piv["GrossProfit"]
        elif "CostOfGoodsSold" in piv:
            gp = rev - piv["CostOfGoodsSold"]
        else:
            gp = float("nan")
        out = pd.DataFrame({"營收": rev, "毛利率": (gp / rev * 100).round(2)})
        out.index = pd.to_datetime(out.index).to_period("Q").astype(str)
        out = out[out["營收"] > 0]
        return out.rename_axis("季度").reset_index().tail(12)
    except Exception:
        return None


# ───────────────────────── 指數（大盤 / 美股） ─────────────────────────
@st.cache_data(ttl=TTL)
def tw_indices() -> dict:
    """{'台股加權指數': Series, '櫃買指數': Series}，近 30 個交易日收盤。"""
    out = {}
    for code, name in (("TAIEX", "台股加權指數"), ("TPEx", "櫃買指數")):
        try:
            d = _dl().taiwan_stock_daily(stock_id=code, start_date=_since(70))
            out[name] = d.set_index(pd.to_datetime(d["date"]))["close"].astype(float).tail(30)
        except Exception:
            pass
    return out


US_SYMBOLS = {
    "費城半導體": "^SOX", "那斯達克": "^IXIC", "標普500": "^GSPC",
    "美元指數": "DX-Y.NYB", "VIX": "^VIX", "10年期殖利率": "^TNX",
}


@st.cache_data(ttl=TTL)
def us_indices() -> dict:
    """美股六大指標近 30 日收盤；yfinance 未安裝或失敗時回傳空 dict。"""
    try:
        import yfinance as yf
        raw = yf.download(list(US_SYMBOLS.values()), period="3mo", progress=False, threads=False, auto_adjust=True)["Close"]
    except Exception:
        return {}
    out = {}
    for name, sym in US_SYMBOLS.items():
        if sym in raw:
            s = raw[sym].dropna().tail(30)
            if sym == "^TNX" and not s.empty and s.iloc[-1] > 20:  # 舊版 yfinance 為 殖利率×10
                s = s / 10
            if len(s) >= 2:
                out[name] = s
    return out


def index_stats(series_map: dict) -> str:
    """把指數序列轉成給 AI 引用的精確數字。"""
    lines = []
    for name, s in series_map.items():
        lo, hi, last, first = s.min(), s.max(), s.iloc[-1], s.iloc[0]
        pos = (last - lo) / (hi - lo) * 100 if hi > lo else 50
        lines.append(f"{name}：最新 {last:.2f}，30日變化 {(last / first - 1) * 100:+.2f}%，"
                     f"區間高 {hi:.2f}／低 {lo:.2f}，位於區間 {pos:.0f}%")
    return "\n".join(lines) or "（系統無法取得指數數據，請自行搜尋並標明來源日期）"
