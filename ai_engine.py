"""AI 層：單一 Gemini 呼叫函式 + 三份報告的 prompt。"""
import datetime as dt
import re
import time

import pandas as pd
import requests

from data_loader import secret

# 使用 Google 官方的 latest 別名，模型改版時不必改程式；可在 secrets 用 GEMINI_MODELS="a,b,c" 覆寫
DEFAULT_MODELS = ["gemini-pro-latest", "gemini-flash-latest", "gemini-2.5-flash"]
API = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

COMMON_RULES = """
寫作規範：
- 繁體中文、Markdown、全部使用阿拉伯數字，百分比保留 2 位小數。
- 客觀專業的第三人稱，不使用「你」，不寫死年份（用「最新」「近期」）。
- 禁用投資指示用語（建議、應該、必須、推薦、買入、賣出、進場、出場、加碼、減碼），
  改用「若欲操作」「可考慮」「值得觀察」「參考」，並註明僅供研究參考、非投資建議。
- 判斷多空時，不可只看漲跌正負就下「強勢／弱勢」結論，必須同時說明：相對區間高低位置、動能是否減弱或止穩，
  並從以下語彙擇一定性：續強格局、衝高乏力／留意拉回、創高但動能背離、續弱格局、跌深有撐／醞釀反彈、破底加速、區間整理／方向未明。
"""


def _models():
    custom = secret("GEMINI_MODELS")
    return [m.strip() for m in custom.split(",") if m.strip()] if custom else DEFAULT_MODELS


def _clean(text: str) -> str:
    text = re.sub(r"\[\s*cite[^\]]*\]", "", text, flags=re.I)  # 移除 grounding 引用標記
    return re.sub(r"[ \t]+([。，、；：])", r"\1", text).strip()


def gemini(api_key: str, prompt: str, search: bool = True, max_tokens: int = 32768):
    """依序嘗試模型；429/5xx 重試，其他錯誤換下一個模型。回傳 (文字, 模型名) 或拋出 RuntimeError。"""
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"maxOutputTokens": max_tokens}}  # Gemini 3 官方建議維持預設溫度
    if search:
        payload["tools"] = [{"google_search": {}}]
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}  # 金鑰放 header，不進網址

    errors = []
    for model in _models():
        for attempt in range(3):
            try:
                r = requests.post(API.format(model), headers=headers, json=payload, timeout=240)
            except requests.RequestException as e:
                errors.append(f"{model}: {e}")
                break
            if r.ok:
                cands = r.json().get("candidates", [])
                parts = cands[0].get("content", {}).get("parts", []) if cands else []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                if len(text) > 200:
                    return _clean(text), model
                errors.append(f"{model}: 回應過短或被截斷")
                break
            if r.status_code in (429, 500, 503) and attempt < 2:
                m = re.search(r"retry in ([\d.]+)s", r.text)
                time.sleep(min(15, float(m.group(1)) if m else 4 * (attempt + 1)))
                continue
            errors.append(f"{model}: HTTP {r.status_code} {r.text[:150]}")
            break
    raise RuntimeError("所有模型皆失敗：\n" + "\n".join(errors[-4:]))


# ───────────────────────── 個股 ─────────────────────────
def _kbar(o, h, l, c):
    rng, body = h - l, abs(c - o)
    if rng <= 0 or rng / o < 0.003:
        return "一字線"
    up = h - max(o, c)
    dn = min(o, c) - l
    red = c >= o
    if body / rng < 0.05:
        return "T字線" if up < rng * 0.1 else "倒T線" if dn < rng * 0.1 else "十字線"
    if up > body * 2 and dn < body * 0.3:
        return "倒鎚" + ("紅K" if red else "黑K")
    if dn > body * 2 and up < body * 0.3:
        return ("紅K" if red else "黑K") + "鎚子"
    if (up + dn) / rng <= 0.2:
        pct = body / o * 100
        size = "大" if pct >= 7 else "中" if pct >= 3 else "小"
        return size + ("紅K" if red else "黑K")
    return "紡錘" + ("紅K" if red else "黑K")


def _stock_context(df, rev, q, holding):
    cols = [c for c in ["date", "open", "high", "low", "close", "volume", "MA20", "MA100", "外資", "投信", "融資餘額"] if c in df]
    t = df[cols].tail(30).copy()
    t["K線"] = [_kbar(*r) for r in df[["open", "high", "low", "close"]].tail(30).itertuples(index=False)]
    t["date"] = t["date"].dt.strftime("%m-%d")
    t = t.round(2)

    last, hi, lo = df["close"].iloc[-1], df["high"].max(), df["low"].min()
    ctx = [f"## 近{len(df)}日區間：高 {hi:.2f}／低 {lo:.2f}，現價 {last:.2f} 位於區間 {(last - lo) / (hi - lo) * 100:.0f}%"]
    if pd.notna(df["MA240"].iloc[-1]):
        ctx.append(f"年線 MA240 {df['MA240'].iloc[-1]:.2f}，乖離 {(last / df['MA240'].iloc[-1] - 1) * 100:+.2f}%")
    ctx.append("## 近30日數據（量/法人/融資單位：張）\n" + t.to_string(index=False))
    if rev is not None:
        r = rev.tail(12).assign(營收億=lambda x: (x["營收"] / 1e8).round(2), 年增率=lambda x: x["年增率"].round(2))
        r["日期"] = r["日期"].dt.strftime("%Y-%m")
        ctx.append("## 月營收（億元／年增率%）\n" + r[["日期", "營收億", "年增率"]].to_string(index=False))
    if q is not None:
        qq = q.tail(8).assign(營收億=lambda x: (x["營收"] / 1e8).round(2))
        ctx.append("## 季營收（億元）與毛利率%\n" + qq[["季度", "營收億", "毛利率"]].to_string(index=False))
    if holding is not None and len(holding) >= 8:
        h = holding["大戶持股比例"].tail(8)
        ctx.append(f"## 大戶(>400張)持股比：近8週 {h.iloc[0]:.2f}% → {h.iloc[-1]:.2f}%（{h.iloc[-1] - h.iloc[0]:+.2f} 個百分點）")
    return "\n\n".join(ctx)


def stock_report(api_key, stock_id, name, df, rev=None, q=None, holding=None):
    prompt = f"""你是台股資深分析師，請針對「{stock_id} {name}」撰寫診斷報告。
下方為系統提供的精確數據，數字以此為準；另請用 Google 搜尋補充最新財報、法人目標價、重大訊息與產業動態（標註來源日期）。

{_stock_context(df, rev, q, holding)}

請依序輸出以下章節（總字數 2200 字以上，每章都要完整）：

### 第一章：K線型態
用「→」串連最近 1-3 日 K 線欄位的型態演變並給一句結論；對照長期區間位置判斷是高檔反轉、低檔止穩或區間中段；
型態可靠度 1-5 分。不要解釋型態定義。

### 第二章：均線與趨勢
只分析 MA20（短期）與 MA100（中期）：排列（多頭／空頭／多箱：價在MA100上MA20下／空箱：價在MA20上MA100下）、乖離、轉折跡象、支撐壓力。
可用一句話帶到年線位置，不可提其他均線。

### 第三章：籌碼
只依表格中的外資、投信、融資實際數字判斷近30日累計與方向，以及大戶持股比趨勢。
基本面利多不代表籌碼流入；若兩者方向不一致，須明確寫出分歧。資料缺漏就略過，不得虛構。

### 第四章：產業與基本面
公司定位與競爭優勢、產業景氣、題材催化劑、財務體質（必須引用上方月營收／季營收數字；金融股不談毛利率）、法人觀點。

### 第五章：操作策略參考
多空方向與時間軸、支撐／壓力／停損參考價、積極型與保守型思路、編號列出風險因子。

### 診斷結語
一張 Markdown 表格「| 面向 | 優勢 ✅ | 劣勢 ❌ |」（技術、籌碼、基本、產業、風險，各 15 字內），
再一張「| 類型 | 價位 | 說明 |」（積極入場參考、保守入場參考、短期目標、中期目標、止損參考）。
{COMMON_RULES}"""
    return gemini(api_key, prompt)


# ───────────────────────── 大盤 ─────────────────────────
def tw_market_report(api_key, stats: str):
    today = dt.date.today().strftime("%Y-%m-%d")
    prompt = f"""今天是 {today}。請搜尋最新資訊，撰寫「台股股市全動態」報告（1400 字以上）。
指數數值以系統數據為準，搜尋只用來補充原因：
{stats}

### 第一節：大盤近30日技術結構
加權與櫃買指數的區間位置、動能、量能、兩者背離或共振，以及與美股／亞股的連動。最後寫【綜合結論】。

### 第二節：投資人氛圍與籌碼
全市場融資融券、外資投信整體買賣超、台指期未平倉與 Put/Call、成交量熱度（不分析個股或類股）。
最後寫【氛圍研判】並說明與第一節是否一致。

### 第三節：總結與下月展望
區間摘要、情緒定性、下月關鍵事件、風險提示。
{COMMON_RULES}"""
    return gemini(api_key, prompt, max_tokens=16384)


def us_market_report(api_key, stats: str):
    today = dt.date.today().strftime("%Y-%m-%d")
    prompt = f"""今天是 {today}。請搜尋最新資訊，撰寫「美股總經雷達」報告（2000 字以上，五節都要完整）。
指數數值以系統數據為準，搜尋只用來補充原因：
{stats}

### 第一節：六大指標快覽
費半、那斯達克、標普500、美元指數、VIX（15以下樂觀／15-25中性／25-35警戒／35以上恐慌）、10年期殖利率，
每項用「指標：數值 → 解讀」格式。最後寫【綜合結論】：指標間共振或背離、最值得警惕的訊號組合。

### 第二節：總經事件
最新 CPI 與核心 CPI、非農、失業率、Fed 利率區間與 FedWatch 預期（標明公布日期）。最後寫【綜合結論】說明政策鬆緊預期。

### 第三節：熱度與恐慌評分
一張表「| 指標 | 數值 | 訊號 | 解讀 |」，含上述指標加「市場熱絡度(1-10)」「恐慌程度(1-10)」，每格都要填實際內容。
表後說明目前是風險偏好或風險規避。

### 第四節：對台股的影響
外資動向、費半對台積電等半導體的牽引、美元與新台幣匯率、下月重大事件（不使用表格）。

### 第五節：總結
整體定性、【攻勢型】或【守勢型】裁定與依據、1-3 項最大風險、台股可關注方向（僅供研究參考）。
{COMMON_RULES}"""
    return gemini(api_key, prompt, max_tokens=16384)
