import re

import streamlit as st

import ai_engine as ai
import chart_plotter as cp
import data_loader as dl

st.set_page_config(page_title="台股 AI 分析平台", page_icon="📈", layout="wide")
st.markdown("<style>h3{color:#5b4b9e} [data-testid=stMetricValue]{font-size:1.5rem}</style>",
            unsafe_allow_html=True)

# ───────────────────────── 側邊欄 ─────────────────────────
with st.sidebar:
    st.title("📈 台股 AI 分析")
    api_key = dl.secret("GEMINI_API_KEY")
    if api_key:
        st.caption("🔑 已從 secrets 載入 Gemini API Key")
    else:
        api_key = st.text_input("Gemini API Key", type="password",
                                help="永久保存：在 .streamlit/secrets.toml 加入 GEMINI_API_KEY = \"...\"")

    with st.form("query"):  # 用表單：調整參數不會每動一下就重跑
        stock_id = st.text_input("股票代碼", "2330").strip()
        days = st.slider("分析天數", 60, 400, 250, 10)
        adjusted = st.toggle("還原K線（除權息調整）", True)
        mas = st.multiselect("均線", list(cp.MA_COLORS), ["MA20", "MA100"])
        st.form_submit_button("🔍 開始分析", type="primary", width="stretch")
    st.caption("資料：FinMind（設定 FINMIND_TOKEN 可提高額度）｜AI 內容僅供研究參考，非投資建議")


def show(fig):
    if fig is not None:
        st.plotly_chart(fig, theme=None, config=cp.CONFIG)


# AI 報告每段文字的「最後一句」通常就是該段落的結論／重點句，自動加上底色方便快速抓重點。
# 不依賴 AI 是否乖乖輸出特定格式的標記（Gemini 不一定每次都照 prompt 格式走），
# 表格、標題、條列式清單則跳過，避免破壞排版。
_LIST_LINE = re.compile(r"^\s*(?:\d+[.、]|[-*•])\s")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？])")
_BRACKET_TAG = re.compile(r"【([^】]{1,12})】")


def _highlight(text: str) -> str:
    out = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        lines = stripped.splitlines()
        is_prose = bool(stripped) and not stripped.startswith(("#", "|", ">")) \
            and not any(_LIST_LINE.match(ln) for ln in lines)
        if is_prose:
            sentences = [s for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]
            last = sentences[-1] if sentences else ""
            # 只標最後一句「完整的句子」（以句號／驚嘆號／問號結尾），
            # 排除單純帶冒號的小標題（例如「題材催化劑：」）誤被當成結論。
            if last.rstrip().endswith(("。", "！", "？")) and len(last.strip()) >= 6:
                head = "".join(sentences[:-1])
                last = last.lstrip().replace("[", "［").replace("]", "］")  # 避免破壞顏色標記語法
                last = _BRACKET_TAG.sub(r"**【\1】**", last)  # 若 AI 有寫【...】標記，一併加粗
                block = head + (" " if head and not head.endswith((" ", "\n")) else "") + f":orange-background[{last}]"
        out.append(block)
    return "\n\n".join(out)


def ai_block(key, label, run):
    """按鈕觸發 AI 並存進 session_state；之後的重跑不會再花 API 額度。"""
    if st.button(label, key=f"btn_{key}", type="primary", disabled=not api_key,
                 help=None if api_key else "請先輸入 Gemini API Key"):
        with st.spinner("AI 分析中（約 1~3 分鐘）..."):
            try:
                st.session_state[key] = run()
            except Exception as e:
                st.session_state[key] = (f"❌ {e}", None)
    if key in st.session_state:
        text, model = st.session_state[key]
        st.markdown(_highlight(text))
        if model:
            st.caption(f"模型：{model}")
            st.download_button("⬇️ 下載報告 (.md)", text, file_name=f"{key}.md")


def market_tab(key, fetch, report, cols):
    if st.button("📊 載入近30日指數", key=f"load_{key}"):
        st.session_state[f"idx_{key}"] = fetch()
    series = st.session_state.get(f"idx_{key}")
    if series is None:
        return
    if not series:
        st.warning("目前無法取得指數資料，請稍後再試。")
    grid = st.columns(cols)
    for i, (name, s) in enumerate(series.items()):
        grid[i % cols].metric(name, f"{s.iloc[-1]:,.2f}", f"{(s.iloc[-1] / s.iloc[0] - 1) * 100:+.2f}%",
                              delta_color="inverse", chart_data=s.round(2).tolist(), chart_type="area", border=True)
    ai_block(f"report_{key}", "🤖 產生 AI 報告", lambda: report(api_key, dl.index_stats(series)))


tab_stock, tab_tw, tab_us = st.tabs(["📊 個股分析", "📡 台股大盤", "🌐 美股總經"])

# ───────────────────────── 個股 ─────────────────────────
def stock_tab():
    with st.spinner(f"載入 {stock_id} ..."):
        df, warns = dl.price_data(stock_id, days, adjusted)
    if df is None:
        st.error("查無資料：代碼可能有誤、為興櫃或已下市。")
        return

    name = dl.stock_info(stock_id)["name"]
    last, prev = df.iloc[-1], df.iloc[-2]
    above20, above100 = last["close"] > last["MA20"], last["close"] > last["MA100"]
    trend = {(True, True): "📈 多頭", (False, False): "📉 空頭",
             (False, True): "📊 多箱整理", (True, False): "📊 空箱整理"}[(bool(above20), bool(above100))]

    st.subheader(f"{stock_id} {name}　{trend}")
    c = st.columns(4)
    c[0].metric("收盤", f"{last['close']:.2f}", f"{last['close'] - prev['close']:+.2f} "
                f"({(last['close'] / prev['close'] - 1) * 100:+.2f}%)", delta_color="inverse")
    c[1].metric("成交量(張)", f"{last['volume']:,}")
    c[2].metric("MA20", f"{last['MA20']:.2f}")
    c[3].metric("MA100", f"{last['MA100']:.2f}")
    if "adj_fallback" in warns:
        st.caption("⚠️ 無還原股價資料，改用一般K線")
    if "rate_limit" in warns:
        st.warning("FinMind 額度已用盡，籌碼資料可能不完整，請稍後再試或設定 FINMIND_TOKEN。")

    show(cp.price_chart(df, f"{stock_id} {name}（{'還原' if adjusted else '一般'}K線）", mas))

    st.markdown("### 💰 籌碼")
    show(cp.chip_chart(df))
    holding = dl.holding_ratio(stock_id)
    if holding is not None:
        show(cp.holding_chart(holding))

    st.markdown("### 📊 營收")
    rev, q = dl.monthly_revenue(stock_id), dl.quarterly(stock_id)
    left, right = st.columns(2)
    with left:
        if rev is not None:
            show(cp.revenue_chart(rev))
        else:
            st.info("查無月營收資料")
    with right:
        if q is not None:
            show(cp.quarterly_chart(q))
        else:
            st.info("查無季報資料")

    st.markdown("### 🤖 AI 深度解盤")
    ai_block(f"stock_{stock_id}_{days}", "🤖 產生 AI 深度解盤",
             lambda: ai.stock_report(api_key, stock_id, name, df, rev, q, holding))


with tab_stock:
    stock_tab()

# ───────────────────────── 大盤 / 美股 ─────────────────────────
with tab_tw:
    st.caption("加權、櫃買指數近30日結構 ＋ AI 即時搜尋投資人氛圍與籌碼氣氛")
    market_tab("tw", dl.tw_indices, ai.tw_market_report, cols=2)

with tab_us:
    st.caption("費半、那指、標普、美元、VIX、10年期殖利率 ＋ CPI／非農／Fed 解讀與攻守勢研判")
    market_tab("us", dl.us_indices, ai.us_market_report, cols=3)
