import streamlit as st
print('[INFO] main.py patched v9 loaded')
from data_loader import StockDataLoader
from chart_plotter import plot_price_volume_chart, plot_chip_chart, plot_margin_chart, plot_revenue_chart, plot_quarterly_chart, plot_holding_ratio_chart
from ai_engine import analyze_stock_trend, generate_quick_summary, analyze_market_flow, analyze_us_market, get_us_index_charts, get_tw_index_charts
import base64
from pathlib import Path
import pandas as pd
import re

def _quick_summary_line(df: pd.DataFrame, full_name: str) -> str:
    """K線上方摘要：收盤固定 2 位小數（避免 23.35000038 這種浮點顯示）"""
    if df is None or df.empty or 'close' not in df.columns:
        return full_name
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    try:
        close = float(latest['close'])
    except Exception:
        close = float(pd.to_numeric(latest.get('close', 0), errors='coerce') or 0)
    try:
        prev_close = float(prev['close'])
    except Exception:
        prev_close = float(pd.to_numeric(prev.get('close', close), errors='coerce') or close)

    chg = close - prev_close
    chg_pct = (chg / prev_close * 100.0) if prev_close else 0.0

    vol = 0
    if 'volume' in df.columns:
        try:
            vol = int(round(float(latest.get('volume', 0))))
        except Exception:
            vol = int(pd.to_numeric(latest.get('volume', 0), errors='coerce') or 0)

    return f"{full_name} 收盤：{close:.2f} ({chg:+.2f} / {chg_pct:+.2f}%) | 量 {vol:,d} 張"


def _highlight_ai_report(md: str) -> str:
    """
    個股分析報告渲染器 —— 與「股市全動態」「美股總經雷達」統一為白底／淺色卡片風格。
    顏色系統統一為：標題紫(#5b4b9e)、欄位標籤橘(#c76a00)、正向紅(#d62839)、負向綠(#2a9d3e)、
    本文深灰(#333333)，不再使用大量雜色。
    """
    if not isinstance(md, str):
        return md

    md = md.replace('\r\n', '\n').replace('\r', '\n')

    out_lines = []
    in_table = False
    in_risk_section = False
    header_cells_count = 0

    for raw in md.split('\n'):
        line = raw.strip()

        if line == "":
            if in_table:
                out_lines.append("</tbody></table></div>")
                in_table = False
            out_lines.append('<div style="height:8px"></div>')
            continue

        # 破損的分隔線／純破折號行：忽略
        if re.match(r'^[\-_—–]{3,}$', line.replace(' ', '')):
            continue

        # 1) 第X章標題
        _chapter_m = re.match(r'^#{0,6}\s*\**\s*(第[一二三四五]章[^*\n]*)\**\s*$', line)
        if _chapter_m:
            in_risk_section = False
            title = _chapter_m.group(1).strip().replace('**', '')
            out_lines.append(
                f"<div style='font-size:22px;font-weight:900;color:#5b4b9e;"
                f"border-bottom:2px solid #b8a9e8;padding:10px 0 8px;"
                f"margin:24px 0 14px'>{title}</div>"
            )
            continue

        # 1c) 診斷結語標題
        _conclusion_m = re.match(r'^#{0,6}\s*\**\s*(診斷結語[^*\n]*)\**\s*$', line)
        if _conclusion_m:
            in_risk_section = False
            title = _conclusion_m.group(1).strip().replace('**', '')
            out_lines.append(
                f"<div style='font-size:22px;font-weight:900;color:#5b4b9e;"
                f"border-bottom:2px solid #b8a9e8;padding:10px 0 8px;"
                f"margin:28px 0 14px'>📋 {title}</div>"
            )
            continue

        # 風險提示：數字編號 → 卡片樣式
        if in_risk_section:
            m_risk = re.match(r'^(\d+)[.、]\s*(.+)$', line)
            if m_risk:
                num = m_risk.group(1)
                content = m_risk.group(2).strip().replace('**', '')
                out_lines.append(
                    f"<div style='display:flex;gap:12px;margin:8px 0;padding:10px 16px;"
                    f"background:#fff7f0;border-left:3px solid #c76a00;border-radius:4px'>"
                    f"<span style='color:#c76a00;font-weight:900;font-size:16px;min-width:20px'>{num}</span>"
                    f"<span style='color:#333333;font-size:16px;line-height:1.8'>{content}</span></div>"
                )
                continue

        # 表格行
        if raw.strip().startswith('|'):
            cells = [c.strip() for c in raw.strip().strip('|').split('|')]
            if all(re.match(r'^[-:]+$', c.replace(' ', '')) for c in cells if c):
                continue
            if all(re.match(r'^[\-_—–]+$', c.replace(' ', '')) for c in cells if c):
                continue

            is_header = any(kw in raw for kw in ['✅', '❌', '面向', '類型', '優勢', '劣勢'])
            if is_header:
                if in_table:
                    out_lines.append("</tbody></table></div>")
                td_html = ''.join(
                    f"<th style='padding:10px 14px;border:1px solid #e3ddf5;"
                    f"background:#f4f1fc;color:#5b4b9e;font-size:15px;"
                    f"font-weight:800;text-align:center'>{c}</th>"
                    for c in cells
                )
                out_lines.append(
                    "<div style='overflow-x:auto;margin:16px 0'>"
                    "<table style='width:100%;border-collapse:collapse;background:#ffffff'>"
                    f"<thead><tr>{td_html}</tr></thead><tbody>"
                )
                in_table = True
                header_cells_count = 0
            else:
                row_bg = "#faf9ff" if header_cells_count % 2 == 0 else "#ffffff"
                td_html = ''
                for i, c in enumerate(cells):
                    color = '#333333'
                    if i == 0:
                        color = '#5b4b9e'
                    elif '❌' in c or any(neg in c for neg in ['劣勢', '風險', '壓力', '下跌', '賣壓']):
                        color = '#2a9d3e'
                    elif '✅' in c or any(pos in c for pos in ['優勢', '支撐', '上漲', '買盤']):
                        color = '#d62839'
                    td_html += (
                        f"<td style='padding:9px 14px;border:1px solid #ece9f7;"
                        f"color:{color};font-size:15px;line-height:1.7'>{c}</td>"
                    )
                out_lines.append(f"<tr style='background:{row_bg}'>{td_html}</tr>")
                header_cells_count += 1
            continue

        if in_table:
            out_lines.append("</tbody></table></div>")
            in_table = False

        # 一般 Markdown 標題 # ~ ######
        m1 = re.match(r'^(#{1,6})\s*(.+)$', line)
        if m1:
            title = m1.group(2).strip().replace('**', '')
            out_lines.append(
                f"<div style='font-size:19px;font-weight:900;color:#8a4fae;"
                f"background:#f8f1fb;border-left:4px solid #c565de;"
                f"padding:10px 16px;margin:18px 0 10px;border-radius:6px'>{title}</div>"
            )
            continue

        # **副標題**（單行）
        m2 = re.match(r'^\*\*(.+?)\*\*\s*[:：]*\s*$', line)
        if m2:
            title = m2.group(1).strip().replace('**', '')
            out_lines.append(f"<div style='font-size:17px;font-weight:800;color:#c76a00;margin:14px 0 6px'>{title}</div>")
            continue

        # 第五章固定副標題
        chapter5_subtitles = ['多空方向', '綜合評分依據', '關鍵價位', '積極型操作思路', '保守型操作思路', '風險提示']
        if any(line.strip() == subtitle for subtitle in chapter5_subtitles):
            if line.strip() == '風險提示':
                in_risk_section = True
            else:
                in_risk_section = False
            out_lines.append(f"<div style='font-size:17px;font-weight:800;color:#c76a00;margin:14px 0 6px'>{line.strip()}</div>")
            continue

        # 小副標題（無**包圍）
        small_subtitle_patterns = [
            '營收趨勢', '年增率變化', '營收高峰', '營收低谷', '動能評估',
            '季營收變化', '季毛利率趨勢', '成本控制能力', '關聯性分析',
            '技術面', '籌碼面', '基本面',
            '第一支撐位', '第二支撐位', '第一壓力位', '第二壓力位', '止損價位',
            '技術性修正風險', '籌碼面不穩定', '基本面壓力', '產業競爭'
        ]
        line_clean = line.strip().rstrip('：:')
        if line_clean in small_subtitle_patterns:
            out_lines.append(
                f"<div style='margin:12px 0 6px 0'><span style='color:#8a4fae;font-weight:800;font-size:16px'>▸ {line_clean}</span></div>"
            )
            continue

        # 趨勢定義為「xxx」→ 強調
        if '趨勢定義為' in line and '「' in line and '」' in line:
            line = re.sub(r'趨勢定義為\s*「([^」]+)」', r'趨勢定義為「<span style="color:#5b4b9e;font-weight:900">\1</span>」', line)
            out_lines.append(f"<div style='color:#333333;font-size:16px;line-height:1.9;margin:5px 0'>{line}</div>")
            continue

        # 項目符號：* **label** 內容
        m_bullet = re.match(r'^\*\s+\*\*([^*]+)\*\*\s*[:：]?\s*(.*)$', line)
        if m_bullet:
            label = m_bullet.group(1).strip()
            content = m_bullet.group(2).strip()
            body = f"<span style='color:#5b4b9e;font-weight:800'>{label}</span>" + (f"：{content}" if content else "")
            out_lines.append(f"<div style='margin:8px 0 6px 16px;color:#333333;font-size:16px;line-height:1.8'>{body}</div>")
            continue

        # 巢狀項目（- / *）
        m_nested = re.match(r'^\s*[-*]\s+(.+)$', line)
        if m_nested:
            content = m_nested.group(1).strip()
            out_lines.append(
                f"<div style='margin:6px 0 6px 20px;color:#333333;font-size:16px;line-height:1.8'>"
                f"<span style='color:#8a4fae;margin-right:8px'>▸</span>{content}</div>"
            )
            continue

        # 欄位名：內容
        m3 = re.match(r'^(•\s*)?([^：]{2,18})(：)(.*)$', line)
        if m3:
            k = m3.group(2).strip()
            rest = m3.group(4).strip()
            out_lines.append(
                f"<div style='color:#333333;font-size:16px;line-height:1.9;margin:5px 0'>"
                f"<span style='color:#c76a00;font-weight:800'>{k}</span>：{rest}</div>"
            )
            continue

        # 一般段落：統一精簡的關鍵字上色（紅漲綠跌 + 少量重點詞）
        line2 = raw.strip()

        # K線型態：紅K系→紅、黑K系→綠，其餘型態統一紫色標示，不再逐一配色
        for pattern in ['大紅K', '中紅K', '小紅K', '紡錘紅K', '倒鎚紅K', '紅K鎚子']:
            line2 = re.sub(f'({pattern})', r"<span style='color:#d62839;font-weight:800'>\1</span>", line2)
        for pattern in ['大黑K', '中黑K', '小黑K', '紡錘黑K', '倒鎚黑K', '黑K鎚子']:
            line2 = re.sub(f'({pattern})', r"<span style='color:#2a9d3e;font-weight:800'>\1</span>", line2)
        for pattern in ['墓碑線', '吊人線', '十字線', 'T字線', '倒T線', '一字線']:
            line2 = re.sub(f'({pattern})', r"<span style='color:#8a4fae;font-weight:800'>\1</span>", line2)

        if not re.search(r'第[一二三四五]章', line2):
            for kw in ['技術面', '籌碼面', '基本面']:
                line2 = re.sub(kw, f"<span style='color:#8a4fae;font-weight:800'>{kw}</span>", line2)

        line2 = re.sub(r'空箱', r"<span style='color:#2a9d3e;font-weight:800'>空箱</span>", line2)
        line2 = re.sub(r'多箱', r"<span style='color:#d62839;font-weight:800'>多箱</span>", line2)
        line2 = re.sub(r'外資', r"<span style='color:#8a4fae;font-weight:800'>外資</span>", line2)
        line2 = re.sub(r'投信', r"<span style='color:#8a4fae;font-weight:800'>投信</span>", line2)
        line2 = re.sub(r'MA100', r"<strong style='color:#5b4b9e'>MA100</strong>", line2)
        line2 = re.sub(r'MA20', r"<strong style='color:#5b4b9e'>MA20</strong>", line2)

        line2 = re.sub(r'買超', r"<span style='color:#d62839;font-weight:800'>買超</span>", line2)
        line2 = re.sub(r'賣超', r"<span style='color:#2a9d3e;font-weight:800'>賣超</span>", line2)
        line2 = re.sub(r'(多方|多頭|上漲|突破|支撐)', r"<span style='color:#d62839;font-weight:700'>\1</span>", line2)
        line2 = re.sub(r'(?<!多)(空方|空頭|下跌|跌破|壓力)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", line2)

        # 負數（%／元／億／千元）統一綠色，正數統一紅色，其餘一律維持本文深灰不特別上色
        line2 = re.sub(r'(-\d+(?:\.\d+)?%)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", line2)
        line2 = re.sub(r'(?<![->])(\+\d+(?:\.\d+)?%)', r"<span style='color:#d62839;font-weight:700'>\1</span>", line2)

        out_lines.append(f"<div style='color:#333333;font-size:16px;line-height:1.9;margin:5px 0'>{line2}</div>")

    if in_table:
        out_lines.append("</tbody></table></div>")
    return '\n'.join(out_lines)

st.set_page_config(
    page_title="台股AI分析平台", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded" 
)

# 自定義CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #fafaff 0%, #f6f4fb 100%);
}

.ai-report{font-size:16px;line-height:1.9;}
.ai-report code{font-size:0.95em;}

    /* 側邊欄整體風格 */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e3ddf5;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #5b4b9e;
    }

    /* 側邊欄Logo樣式 */
    .sidebar-logo {
        text-align: center;
        padding: 15px 0;
        margin-bottom: 15px;
        border-bottom: 1px solid #e3ddf5;
    }
    .sidebar-logo img {
        width: 150px;
        height: auto;
        border-radius: 10px;
    }

    /* 主要按鈕統一為科技紫，取代 Streamlit 預設紅色 */
    .stButton > button[kind="primary"] {
        background-color: #5b4b9e;
        border-color: #5b4b9e;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4a3d82;
        border-color: #4a3d82;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #5b4b9e;
        color: #5b4b9e;
    }

    /* 指數走勢卡片：預設多欄並排，手機窄螢幕自動改為單欄置中垂直排列 */
    .idx-chart-grid {
        display: grid;
        grid-template-columns: repeat(var(--cols, 3), 1fr);
        gap: 14px;
    }
    @media (max-width: 640px) {
        .idx-chart-grid {
            grid-template-columns: 1fr !important;
        }
        .idx-chart-grid > div {
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

# ========== Logo 放在最上方（sidebar 開始之前）==========
logo_path = Path(__file__).parent / "YT.png"
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.sidebar.markdown(f"""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """, unsafe_allow_html=True)

st.sidebar.title("⚙️ 分析設定")

with st.sidebar.expander("🔑 AI 設定", expanded=True):

    # ── 從 secrets.toml 自動載入（若有設定）────────────────────────────────
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    # ── 用 session_state 跨次保存（同一瀏覽器 session 不需重輸）────────────
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = _secret_key  # 優先用 secrets

    def _on_api_key_change():
        st.session_state["gemini_api_key"] = st.session_state["_api_key_input"]

    api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state["gemini_api_key"],
        type="password",
        key="_api_key_input",
        on_change=_on_api_key_change,
        help="輸入後自動記憶，重新整理頁面不需重輸。若想永久儲存請參考下方說明。"
    )
    # 確保變數同步
    api_key = st.session_state["gemini_api_key"]

    if api_key:
        st.success("✅ API Key 已載入", icon="🔑")
        if st.button("🗑️ 清除 API Key", use_container_width=True):
            st.session_state["gemini_api_key"] = ""
            st.rerun()
    else:
        st.info("💡 **永久儲存方法：**\n\n"
                "在程式資料夾建立 `.streamlit/secrets.toml`，\n"
                "加入一行：\n\n"
                "`GEMINI_API_KEY = \"你的Key\"`\n\n"
                "之後啟動就會自動帶入，不需手動輸入。")

with st.sidebar.expander("📊 查詢參數", expanded=True):
    stock_id = st.text_input("股票代碼", value="2330", help="例如：2330, 2317")
    days = st.slider("分析天數", min_value=60, max_value=400, value=250, step=10)
    
    # K線類型（預設還原K線）
    st.markdown("**K線類型(預設還原)**")
    use_normal = st.checkbox("使用一般K線（未還原）", value=False, 
                             help="勾選此項將顯示實際交易價格（有除權息跳空）\n不勾選則使用還原K線（消除除權息影響）")
    use_adjusted = not use_normal  # 反轉邏輯
    
    st.markdown("**均線顯示**")
    show_ma_dict = {
        'MA5': st.checkbox("5日線", value=False),
        'MA20': st.checkbox("20日線 (月線)", value=True),
        'MA60': st.checkbox("60日線 (季線)", value=False),
        'MA100': st.checkbox("100日線", value=True),
        'MA120': st.checkbox("120日線 (半年線)", value=False),
        'MA240': st.checkbox("240日線 (年線)", value=False),
    }

run_analysis = st.sidebar.button("🔍 開始分析", type="primary", use_container_width=True)

# ✅ 在 with 區塊外重新讀取 api_key，確保首頁也能正確取得
api_key = st.session_state.get("gemini_api_key", "")


def _render_market_report(md: str) -> str:
    """
    股市全動態專用渲染器 —— 白色／淺色卡片風格（與美股總經雷達一致）。
    """
    import re

    cleaned_lines = []
    prev_was_table = False
    for raw in md.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        is_table = raw.strip().startswith('|')
        if prev_was_table and raw.strip() == '':
            cleaned_lines.append('__TABLE_SPACER__')
        else:
            cleaned_lines.append(raw)
        prev_was_table = is_table

    out = []
    in_table = False
    table_row_count = 0

    for raw in cleaned_lines:
        if raw == '__TABLE_SPACER__':
            continue

        line = raw.strip()

        if line == '':
            if in_table:
                out.append('</tbody></table></div>')
                in_table = False
                table_row_count = 0
            out.append('<div style="height:10px"></div>')
            continue

        # ── 破損的分隔線／純破折號行：直接忽略 ──
        if re.match(r'^[\-_—–]{3,}$', line.replace(' ', '')):
            continue

        # ── 表格行 ──
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            cells = [c for c in cells if c != '']
            if not cells:
                continue
            if all(re.match(r'^[-:]+$', c.replace(' ', '')) for c in cells):
                continue
            if all(re.match(r'^[\-_—–]+$', c.replace(' ', '')) for c in cells if c):
                continue

            if not in_table:
                td_html = ''.join(
                    f"<th style='padding:10px 14px;border:1px solid #e3ddf5;"
                    f"background:#f4f1fc;color:#5b4b9e;font-size:15px;"
                    f"font-weight:800;text-align:center'>{c}</th>"
                    for c in cells
                )
                out.append(
                    "<div style='overflow-x:auto;margin:16px 0'>"
                    "<table style='width:100%;border-collapse:collapse;background:#ffffff'>"
                    f"<thead><tr>{td_html}</tr></thead><tbody>"
                )
                in_table = True
                table_row_count = 1
            else:
                row_bg = "#faf9ff" if table_row_count % 2 == 0 else "#ffffff"
                td_cells = []
                for c in cells:
                    color = '#333333'
                    if '初步布局' in c or '初步流入' in c:
                        color = '#c9990b'
                    elif '流入' in c or '墊高' in c or '多頭' in c or '強勢' in c or c.startswith('+'):
                        color = '#d62839'
                    elif '流出' in c or '探底' in c or '空頭' in c or '弱勢' in c or c.startswith('-'):
                        color = '#2a9d3e'
                    td_cells.append(
                        f"<td style='padding:9px 14px;border:1px solid #ece9f7;"
                        f"color:{color};font-size:15px;line-height:1.7'>{c}</td>"
                    )
                out.append(f"<tr style='background:{row_bg}'>{''.join(td_cells)}</tr>")
                table_row_count += 1
            continue

        if in_table:
            out.append('</tbody></table></div>')
            in_table = False
            table_row_count = 0

        # ── ### 節標題 ──
        m_h3 = re.match(r'^#{1,3}\s+(.+)$', line)
        if m_h3:
            title = m_h3.group(1).replace('**', '').strip()
            out.append(
                f"<div style='font-size:22px;font-weight:900;color:#5b4b9e;"
                f"border-bottom:2px solid #b8a9e8;padding:10px 0 8px;"
                f"margin:24px 0 14px'>{title}</div>"
            )
            continue

        # ── #### 子標題 ──
        m_h4 = re.match(r'^#{4,6}\s+(.+)$', line)
        if m_h4:
            title = m_h4.group(1).replace('**', '').strip()
            out.append(
                f"<div style='font-size:19px;font-weight:900;color:#8a4fae;"
                f"background:#f8f1fb;border-left:4px solid #c565de;"
                f"padding:10px 16px;margin:18px 0 10px;border-radius:6px'>{title}</div>"
            )
            continue

        # ── 綜合結論區塊（【xxx】格式）──
        m_conclusion = re.match(r'^【(.+?)】\s*[:：]?\s*(.*)$', line)
        if m_conclusion:
            label = m_conclusion.group(1).strip()
            content = m_conclusion.group(2).strip()
            out.append(
                f"<div style='margin:12px 0;padding:14px 20px;background:#f4f1fc;"
                f"border:1px solid #5b4b9e;border-left:5px solid #5b4b9e;"
                f"border-radius:8px;line-height:1.9'>"
                f"<span style='color:#5b4b9e;font-weight:900;font-size:16px'>【{label}】</span>"
                + (f" <span style='color:#333333;font-size:16px'>{content}</span>" if content else "")
                + "</div>"
            )
            continue

        # ── **副標題** 單行 ──
        if re.match(r'^\*\*(.+?)\*\*\s*[:：]?\s*$', line):
            title = re.match(r'^\*\*(.+?)\*\*', line).group(1)
            out.append(f"<div style='font-size:17px;font-weight:800;color:#c76a00;margin:14px 0 6px'>{title}</div>")
            continue

        # ── 數字列表 ──
        m_num = re.match(r'^(\d+)[.、]\s+(.+)$', line)
        if m_num:
            content = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", m_num.group(2))
            out.append(
                f"<div style='display:flex;gap:12px;margin:8px 0;padding:10px 16px;"
                f"background:#f4f1fc;border-left:3px solid #5b4b9e;border-radius:4px'>"
                f"<span style='color:#5b4b9e;font-weight:900;font-size:17px;min-width:22px'>{m_num.group(1)}</span>"
                f"<span style='color:#333333;font-size:16px;line-height:1.8'>{content}</span></div>"
            )
            continue

        # ── 子彈列表 ──
        m_bullet = re.match(r'^\s*[-*]\s+(.+)$', line)
        if m_bullet:
            content = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", m_bullet.group(1))
            out.append(
                f"<div style='margin:6px 0 6px 20px;color:#333333;font-size:16px;line-height:1.8'>"
                f"<span style='color:#8a4fae;margin-right:8px'>▸</span>{content}</div>"
            )
            continue

        # ── 個股/指標欄位 ──
        m_field = re.match(r'^(技術面|基本面|籌碼面|參考關注價位|近期催化劑|資金輪動路徑)[:：]\s*(.*)$', line)
        if m_field:
            key_color = {'技術面':'#2c7fb8','基本面':'#2a9d3e','籌碼面':'#8a4fae',
                         '參考關注價位':'#c9990b','近期催化劑':'#c76a00','資金輪動路徑':'#d62839'}.get(m_field.group(1), '#8a4fae')
            out.append(
                f"<div style='margin:8px 0;padding:10px 18px;background:#f9f8fd;"
                f"border-radius:6px;border-left:4px solid {key_color};line-height:1.8'>"
                f"<span style='color:{key_color};font-weight:800;font-size:16px'>{m_field.group(1)}：</span>"
                f"<span style='color:#333333;font-size:16px'>{m_field.group(2)}</span></div>"
            )
            continue

        # ── 一般段落 ──
        para = line
        para = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", para)
        para = para.replace('→', "<span style='color:#8a4fae;font-weight:900'> → </span>")
        para = re.sub(r'買超', "<span style='color:#d62839;font-weight:800'>買超</span>", para)
        para = re.sub(r'賣超', "<span style='color:#2a9d3e;font-weight:800'>賣超</span>", para)
        para = re.sub(r'(上漲|上升|走強)', r"<span style='color:#d62839;font-weight:700'>\1</span>", para)
        para = re.sub(r'(下跌|下滑|走弱)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", para)
        para = re.sub(r'(-\d+(?:\.\d+)?%)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", para)
        para = re.sub(r'(?<![->])(\+\d+(?:\.\d+)?%)', r"<span style='color:#d62839;font-weight:700'>\1</span>", para)
        out.append(f"<div style='color:#333333;font-size:16px;line-height:1.9;margin:5px 0'>{para}</div>")

    if in_table:
        out.append('</tbody></table></div>')
    return '\n'.join(out)


def _render_index_chart_cards(chart_data, title="📈 六大指數近30日走勢一覽", columns=3) -> str:
    """
    以淺色卡片呈現指數的即時走勢小圖（sparkline）+ 最新數值/漲跌幅。
    chart_data 來自 ai_engine.get_us_index_charts() / get_tw_index_charts()；
    若為 None 代表環境缺少 yfinance / matplotlib，改顯示提示訊息。若全部指數都
    抓取失敗，額外顯示第一個錯誤訊息協助排查（常見原因：Yahoo Finance 封鎖雲端主機請求）。
    """
    if not chart_data:
        return (
            "<div style='padding:14px 18px;background:#fff8e1;border:1px solid #ffd54f;"
            "border-radius:10px;color:#8a6d00;font-size:14px;margin:12px 0 20px'>"
            "⚠️ 尚未安裝 <code>yfinance</code> / <code>matplotlib</code>，暫時無法顯示指數走勢圖。"
            "請在環境中執行：<code>pip install yfinance matplotlib</code></div>"
        )

    all_failed = all(item.get("error") for item in chart_data)
    diagnostic_html = ""
    if all_failed:
        first_err = next((item.get("error") for item in chart_data if item.get("error")), "未知錯誤")
        diagnostic_html = (
            "<div style='padding:12px 16px;background:#fff3e0;border:1px solid #ffcc80;"
            "border-radius:10px;color:#8a4b00;font-size:13px;margin-bottom:12px;line-height:1.7'>"
            "⚠️ 指數目前皆無法取得走勢資料，常見原因是資料源暫時限制雲端主機的請求頻率。"
            f"錯誤細節：{first_err}<br>"
            "可稍後按「重新分析」再試一次，或執行 <code>pip install --upgrade yfinance</code> 更新版本。"
            "</div>"
        )

    cards = []
    for item in chart_data:
        unit = item.get("unit", "")
        if item.get("error") or not item.get("img_base64"):
            cards.append(f"""
            <div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;
                        padding:14px 16px;box-shadow:0 1px 6px rgba(90,70,180,0.06)'>
                <div style='font-weight:800;color:#5b4b9e;font-size:14px'>{item['name']}</div>
                <div style='color:#aaa;font-size:12px;margin-top:16px;text-align:center'>資料暫時無法取得</div>
            </div>""")
            continue

        chg = item["change"]
        chg_pct = item["change_pct"]
        latest = item["latest"]
        up = chg >= 0
        color = "#d62839" if up else "#2a9d3e"
        arrow = "▲" if up else "▼"

        cards.append(f"""
        <div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;
                    padding:14px 16px;box-shadow:0 1px 6px rgba(90,70,180,0.06)'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-weight:800;color:#5b4b9e;font-size:14px'>{item['name']}</span>
                <span style='color:#aaa;font-size:11px'>{item['code']}</span>
            </div>
            <img src='data:image/png;base64,{item["img_base64"]}'
                 style='width:100%;height:44px;margin:8px 0;display:block' />
            <div style='display:flex;justify-content:space-between;align-items:baseline'>
                <span style='font-size:20px;font-weight:900;color:#333'>{latest:.2f}{unit}</span>
                <span style='font-size:13px;font-weight:800;color:{color}'>{arrow} {chg:+.2f} ({chg_pct:+.2f}%)</span>
            </div>
        </div>""")

    return (
        "<div style='margin:12px 0 22px'>"
        f"<div style='font-size:15px;font-weight:800;color:#5b4b9e;margin-bottom:10px'>{title}</div>"
        f"{diagnostic_html}"
        f"<div class='idx-chart-grid' style='--cols:{columns}'>{''.join(cards)}</div>"
        "</div>"
    )


# 向下相容別名（既有呼叫點沿用舊函式名稱）
def _render_us_index_chart_cards(chart_data) -> str:
    return _render_index_chart_cards(chart_data, title="📈 六大指數近30日走勢一覽", columns=3)


def _render_us_market_report(md: str) -> str:
    """
    美股總經雷達專用渲染器 —— 白色／淺色卡片風格。
    """
    import re

    cleaned_lines = []
    prev_was_table = False
    for raw in md.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        is_table = raw.strip().startswith('|')
        if prev_was_table and raw.strip() == '':
            cleaned_lines.append('__TABLE_SPACER__')
        else:
            cleaned_lines.append(raw)
        prev_was_table = is_table

    out = []
    in_table = False
    table_row_count = 0
    header_col_count = 0

    for raw in cleaned_lines:
        if raw == '__TABLE_SPACER__':
            continue

        line = raw.strip()

        if line == '':
            if in_table:
                out.append('</tbody></table></div>')
                in_table = False
                table_row_count = 0
                header_col_count = 0
            out.append('<div style="height:10px"></div>')
            continue

        # ── 破損的分隔線／純破折號行（如 AI 誤輸出整排 "----"）：直接忽略 ──
        if re.match(r'^[\-_—–]{3,}$', line.replace(' ', '')):
            continue

        # ── 表格行 ──
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            cells = [c for c in cells if c != '']
            if not cells:
                continue
            # 分隔行（|---|---|）一律略過
            if all(re.match(r'^[-:]+$', c.replace(' ', '')) for c in cells):
                continue
            # 整列都是破折號佔位符（AI 格式跑掉時常見），視為無效資料列，略過
            if all(re.match(r'^[\-_—–]+$', c.replace(' ', '')) for c in cells if c):
                continue

            if not in_table:
                td_html = ''.join(
                    f"<th style='padding:10px 14px;border:1px solid #e3ddf5;"
                    f"background:#f4f1fc;color:#5b4b9e;font-size:15px;"
                    f"font-weight:800;text-align:center'>{c}</th>"
                    for c in cells
                )
                out.append(
                    "<div style='overflow-x:auto;margin:16px 0'>"
                    "<table style='width:100%;border-collapse:collapse;background:#ffffff'>"
                    f"<thead><tr>{td_html}</tr></thead><tbody>"
                )
                in_table = True
                table_row_count = 1
                header_col_count = len(cells)
            else:
                # 若欄位數與表頭不符（AI 偶爾漏欄），仍盡量呈現，不強制丟棄整行
                row_bg = "#faf9ff" if table_row_count % 2 == 0 else "#ffffff"
                td_cells = []
                for c in cells:
                    color = '#333333'
                    if '流入' in c or '多頭' in c or '強勢' in c or '攻勢' in c or c.startswith('+'):
                        color = '#d62839'
                    elif '流出' in c or '空頭' in c or '弱勢' in c or '守勢' in c or c.startswith('-'):
                        color = '#2a9d3e'
                    elif '恐慌' in c or '警戒' in c:
                        color = '#c76a00'
                    elif '樂觀' in c or '貪婪' in c:
                        color = '#d62839'
                    td_cells.append(
                        f"<td style='padding:9px 14px;border:1px solid #ece9f7;"
                        f"color:{color};font-size:15px;line-height:1.7'>{c}</td>"
                    )
                out.append(f"<tr style='background:{row_bg}'>{''.join(td_cells)}</tr>")
                table_row_count += 1
            continue

        if in_table:
            out.append('</tbody></table></div>')
            in_table = False
            table_row_count = 0
            header_col_count = 0

        # ── ### 節標題 ──
        m_h3 = re.match(r'^#{1,3}\s+(.+)$', line)
        if m_h3:
            title = m_h3.group(1).replace('**', '').strip()
            out.append(
                f"<div style='font-size:22px;font-weight:900;color:#5b4b9e;"
                f"border-bottom:2px solid #b8a9e8;padding:10px 0 8px;"
                f"margin:24px 0 14px'>{title}</div>"
            )
            continue

        # ── #### 子標題 ──
        m_h4 = re.match(r'^#{4,6}\s+(.+)$', line)
        if m_h4:
            title = m_h4.group(1).replace('**', '').strip()
            out.append(
                f"<div style='font-size:19px;font-weight:900;color:#8a4fae;"
                f"background:#f8f1fb;border-left:4px solid #c565de;"
                f"padding:10px 16px;margin:18px 0 10px;border-radius:6px'>{title}</div>"
            )
            continue

        # ── 綜合結論區塊（【xxx】格式）──
        m_conclusion = re.match(r'^【(.+?)】\s*[:：]?\s*(.*)$', line)
        if m_conclusion:
            label = m_conclusion.group(1).strip()
            content = m_conclusion.group(2).strip()
            is_verdict = any(kw in label for kw in ['攻勢型', '守勢型', '最終裁定', '整體判斷'])
            bg_color = "#fdecec" if '攻勢' in label else "#eaf7ec" if '守勢' in label else "#f4f1fc"
            border_color = "#d62839" if '攻勢' in label else "#2a9d3e" if '守勢' in label else "#5b4b9e"
            label_color = border_color
            font_size = "18px" if is_verdict else "16px"
            out.append(
                f"<div style='margin:12px 0;padding:14px 20px;background:{bg_color};"
                f"border:1px solid {border_color};border-left:5px solid {border_color};"
                f"border-radius:8px;line-height:1.9'>"
                f"<span style='color:{label_color};font-weight:900;font-size:{font_size}'>"
                f"【{label}】</span>"
                + (f" <span style='color:#333333;font-size:16px'>{content}</span>" if content else "")
                + "</div>"
            )
            continue

        # ── **label**：內容（同行帶正文，如「**費城半導體**：近30日...」）──
        m_bold_inline = re.match(r'^\*\*(.+?)\*\*\s*[:：]\s*(.+)$', line)
        if m_bold_inline:
            label = m_bold_inline.group(1).strip()
            content = m_bold_inline.group(2).strip()
            content = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", content)
            content = re.sub(r'(-\d+(?:\.\d+)?%)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", content)
            content = re.sub(r'(?<![->])(\+\d+(?:\.\d+)?%)', r"<span style='color:#d62839;font-weight:700'>\1</span>", content)
            out.append(
                f"<div style='margin:10px 0;padding:10px 18px;background:#faf7ff;"
                f"border-left:3px solid #e08a3c;border-radius:5px;line-height:1.8'>"
                f"<span style='color:#c76a00;font-weight:800;font-size:16px'>{label}：</span>"
                f"<span style='color:#333333;font-size:16px'>{content}</span></div>"
            )
            continue

        # ── **副標題** 單行（純標題，無正文）──
        if re.match(r'^\*\*(.+?)\*\*\s*[:：]?\s*$', line):
            title = re.match(r'^\*\*(.+?)\*\*', line).group(1)
            out.append(f"<div style='font-size:17px;font-weight:800;color:#c76a00;margin:14px 0 6px'>{title}</div>")
            continue

        # ── 數字列表 ──
        m_num = re.match(r'^(\d+)[.、]\s+(.+)$', line)
        if m_num:
            content = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", m_num.group(2))
            out.append(
                f"<div style='display:flex;gap:12px;margin:8px 0;padding:10px 16px;"
                f"background:#f4f1fc;border-left:3px solid #5b4b9e;border-radius:4px'>"
                f"<span style='color:#5b4b9e;font-weight:900;font-size:17px;min-width:22px'>{m_num.group(1)}</span>"
                f"<span style='color:#333333;font-size:16px;line-height:1.8'>{content}</span></div>"
            )
            continue

        # ── 子彈列表 ──
        m_bullet = re.match(r'^\s*[-*]\s+(.+)$', line)
        if m_bullet:
            content = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", m_bullet.group(1))
            out.append(
                f"<div style='margin:6px 0 6px 20px;color:#333333;font-size:16px;line-height:1.8'>"
                f"<span style='color:#8a4fae;margin-right:8px'>▸</span>{content}</div>"
            )
            continue

        # ── 指標欄位（攻守勢特殊處理）──
        m_field = re.match(r'^(資金流入|資金流出|攻勢型|守勢型|輪動路徑|攻守勢研判|資金輪動路徑)[:：]\s*(.*)$', line)
        if m_field:
            key_color = {
                '資金流入': '#d62839', '攻勢型': '#d62839',
                '資金流出': '#2a9d3e', '守勢型': '#2a9d3e',
                '輪動路徑': '#c9990b', '資金輪動路徑': '#c9990b',
                '攻守勢研判': '#5b4b9e',
            }.get(m_field.group(1), '#8a4fae')
            out.append(
                f"<div style='margin:8px 0;padding:10px 18px;background:#f9f8fd;"
                f"border-radius:6px;border-left:4px solid {key_color};line-height:1.8'>"
                f"<span style='color:{key_color};font-weight:800;font-size:16px'>{m_field.group(1)}：</span>"
                f"<span style='color:#333333;font-size:16px'>{m_field.group(2)}</span></div>"
            )
            continue

        # ── 一般段落 ──
        para = line  # 使用 stripped 版本，避免前導空白導致不顯示
        para = re.sub(r'\*\*(.+?)\*\*', r"<strong style='color:#5b4b9e'>\1</strong>", para)
        para = para.replace('→', "<span style='color:#8a4fae;font-weight:900'> → </span>")
        # 攻守勢關鍵詞
        para = re.sub(r'攻勢型', "<span style='color:#d62839;font-weight:800;background:#fdecec;padding:1px 5px;border-radius:3px'>攻勢型</span>", para)
        para = re.sub(r'守勢型', "<span style='color:#2a9d3e;font-weight:800;background:#eaf7ec;padding:1px 5px;border-radius:3px'>守勢型</span>", para)
        para = re.sub(r'資金流入', "<span style='color:#d62839;font-weight:800'>資金流入</span>", para)
        para = re.sub(r'資金流出', "<span style='color:#2a9d3e;font-weight:800'>資金流出</span>", para)
        para = re.sub(r'買超', "<span style='color:#d62839;font-weight:800'>買超</span>", para)
        para = re.sub(r'賣超', "<span style='color:#2a9d3e;font-weight:800'>賣超</span>", para)
        para = re.sub(r'(上漲|走強|多頭)', r"<span style='color:#d62839;font-weight:700'>\1</span>", para)
        para = re.sub(r'(下跌|走弱|空頭)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", para)
        para = re.sub(r'恐慌', "<span style='color:#c76a00;font-weight:800'>恐慌</span>", para)
        para = re.sub(r'(貪婪|樂觀)', r"<span style='color:#d62839;font-weight:700'>\1</span>", para)
        para = re.sub(r'(-\d+(?:\.\d+)?%)', r"<span style='color:#2a9d3e;font-weight:700'>\1</span>", para)
        para = re.sub(r'(?<![->])(\+\d+(?:\.\d+)?%)', r"<span style='color:#d62839;font-weight:700'>\1</span>", para)
        # 指數名稱高亮
        for idx_name in ['費城半導體', 'SOX', '那斯達克', 'NASDAQ', '標普500', 'S&P\u00a0500', 'S&P 500', 'VIX', '美元指數', 'DXY', '公債殖利率', 'CPI', '非農', 'FED', 'FOMC']:
            para = re.sub(re.escape(idx_name), f"<span style='color:#8a4fae;font-weight:800'>{idx_name}</span>", para)
        out.append(f"<div style='color:#333333;font-size:16px;line-height:1.9;margin:5px 0'>{para}</div>")

    if in_table:
        out.append('</tbody></table></div>')
    return '\n'.join(out)


if run_analysis and stock_id:
    loader = StockDataLoader()
    
    k_type = "一般K線(未還原)" if use_normal else "還原K線"
    with st.spinner(f"🔄 正在載入 {stock_id} 數據 ({k_type})..."):
        df, error, stock_name = loader.get_combined_data(stock_id, days, use_adjusted)
    
    if error:
        st.error(error)
        st.stop()
    
    # 趨勢判斷
    latest = df.iloc[-1]
    trend_status = "盤整/不明"
    if 'MA20' in df.columns and 'MA100' in df.columns:
        price = latest['close']
        ma20 = latest['MA20']
        ma100 = latest['MA100']
        if price > ma20 and price > ma100: trend_status = "📈 多頭格局"
        elif price < ma20 and price < ma100: trend_status = "📉 空頭格局"
        elif price > ma100 and price < ma20: trend_status = "📊 多箱整理"
        elif price < ma100 and price > ma20: trend_status = "📊 空箱整理"

    st.title(f"📊 {stock_id} {stock_name} 趨勢分析 | {trend_status}")
    
    # 顯示 K 線類型
    k_type_display = "一般K線(未還原)" if use_normal else "還原K線"
    k_type_color = "#c9990b" if use_normal else "#2a9d3e"
    st.markdown(
        f"<div style='background-color: #f4f1fc; padding: 10px; border-radius: 5px; "
        f"border-left: 4px solid {k_type_color}; margin-bottom: 20px;'>"
        f"<b style='color: {k_type_color};'>📈 {k_type_display}</b></div>",
        unsafe_allow_html=True
    )
    
    st.info(_quick_summary_line(df, f"{stock_id} {stock_name}"))
    
    # ── 股價 K 線 + 成交量 ──────────────────────────────────────────
    with st.spinner("繪製圖表中..."):
        fig_price = plot_price_volume_chart(df, stock_id, stock_name, show_ma_dict)
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            "padding:12px;margin-bottom:20px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>",
            unsafe_allow_html=True
        )
        # 均線速覽資訊列（比照看盤軟體：MA數值 + 漲跌方向箭頭，顏色與圖表線條一致）
        ma_labels = {'MA5': '5日', 'MA20': '20日', 'MA60': '60日',
                     'MA100': '100日', 'MA120': '120日', 'MA240': '240日'}
        ma_line_colors = {
            'MA5': '#f4a300', 'MA20': '#c2255c', 'MA60': '#7c3aed',
            'MA100': '#0891b2', 'MA120': '#ea580c', 'MA240': '#b91c1c',
        }
        ma_chips = []
        for ma_name, show in show_ma_dict.items():
            if show and ma_name in df.columns:
                valid_ma = df[df[ma_name] > 0]
                if len(valid_ma) >= 2:
                    latest_v = valid_ma[ma_name].iloc[-1]
                    prev_v = valid_ma[ma_name].iloc[-2]
                    up = latest_v >= prev_v
                    arrow = "▲" if up else "▼"
                    arrow_color = "#d62839" if up else "#2a9d3e"
                    label_color = ma_line_colors.get(ma_name, '#5b4b9e')
                    ma_chips.append(
                        f"<span style='margin-right:16px;font-size:14px'>"
                        f"<b style='color:{label_color}'>({ma_labels.get(ma_name, ma_name)})</b> "
                        f"<b style='color:{arrow_color}'>{latest_v:.2f}{arrow}</b></span>"
                    )
        if ma_chips:
            st.markdown(
                f"<div style='padding:4px 8px 10px 8px'>{''.join(ma_chips)}</div>",
                unsafe_allow_html=True
            )
        st.plotly_chart(
            fig_price, 
            use_container_width=True, 
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{stock_id}_{stock_name}_price',
                    'height': 900,
                    'width': 1400,
                    'scale': 2
                }
            },
            key=f'price_{stock_id}'  # keep chart component stable so zoom won't reset
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 外資 / 投信 買賣超 ──────────────────────────────────────────
    st.subheader("💰 法人籌碼動向")
    with st.spinner("繪製籌碼圖表中..."):
        fig_chip = plot_chip_chart(df, stock_id, stock_name)
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            "padding:12px;margin-bottom:20px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(
            fig_chip,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{stock_id}_{stock_name}_chip',
                    'height': 700,
                    'width': 1400,
                    'scale': 2
                }
            },
            key=f'chip_{stock_id}'
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 大戶持股比例（>400張）──────────────────────────────────────
    df_holding, holding_err = loader.get_holding_shares_ratio(stock_id)
    holding_summary = None
    if df_holding is not None and not df_holding.empty:
        with st.spinner("繪製大戶持股比例圖表中..."):
            fig_holding = plot_holding_ratio_chart(df_holding, stock_id, stock_name)
            st.markdown(
                "<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
                "padding:12px;margin-bottom:20px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>",
                unsafe_allow_html=True
            )
            st.plotly_chart(
                fig_holding,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{stock_id}_{stock_name}_holding',
                        'height': 500,
                        'width': 1400,
                        'scale': 2
                    }
                },
                key=f'holding_{stock_id}'
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # 準備一段文字摘要，供 AI 第三章籌碼分析引用（近8週趨勢）
        recent_holding = df_holding.tail(8)
        if len(recent_holding) >= 2:
            chg = recent_holding['大戶持股比例'].iloc[-1] - recent_holding['大戶持股比例'].iloc[0]
            holding_summary = (
                f"大戶持股比例（>400張）近8週由 {recent_holding['大戶持股比例'].iloc[0]:.2f}% "
                f"變化至 {recent_holding['大戶持股比例'].iloc[-1]:.2f}%（變動 {chg:+.2f} 個百分點），"
                f"最新一週為 {df_holding['大戶持股比例'].iloc[-1]:.2f}%"
            )
    elif holding_err:
        st.caption(f"ℹ️ 大戶持股比例：{holding_err}")

    # ── 主力法人15日累計 vs 融資 ────────────────────────────────────
    with st.spinner("繪製主力/融資圖表中..."):
        fig_margin = plot_margin_chart(df, stock_id, stock_name)
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            "padding:12px;margin-bottom:20px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(
            fig_margin,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{stock_id}_{stock_name}_margin',
                    'height': 600,
                    'width': 1400,
                    'scale': 2
                }
            },
            key=f'margin_{stock_id}'
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== 月營收與年增率圖表 ==========
    st.markdown("---")
    st.subheader("📊 月營收與年增率分析")
    
    with st.spinner("載入月營收數據..."):
        df_revenue, rev_error = loader.get_monthly_revenue(stock_id)
    
    if rev_error:
        st.warning(f"⚠️ {rev_error}")
    elif df_revenue is not None and not df_revenue.empty:
        # 顯示最新月營收摘要
        latest_rev = df_revenue.iloc[-1]
        if pd.notna(latest_rev['營收']) and pd.notna(latest_rev['年增率']):
            yoy_icon = "📈" if latest_rev['年增率'] >= 0 else "📉"
            yoy_color = "red" if latest_rev['年增率'] >= 0 else "green"
            st.markdown(
                f"{yoy_icon} 最新月營收：**{int(latest_rev['年'])}年{int(latest_rev['月'])}月** | "
                f"營收 **{latest_rev['營收']/100000000:.2f}** 億元 | "
                f"年增率 <span style='color:{yoy_color}'>**{latest_rev['年增率']:+.2f}%**</span>",
                unsafe_allow_html=True
            )
        
        # 繪製月營收圖表
        fig_revenue = plot_revenue_chart(df_revenue, stock_id, stock_name)
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            "padding:12px;margin-bottom:10px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(
            fig_revenue,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{stock_id}_{stock_name}_revenue',
                    'height': 700,
                    'width': 1400,
                    'scale': 2
                }
            },
            key=f'revenue_{stock_id}'
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== 季營收與季毛利率圖表 ==========
    st.markdown("---")
    
    with st.spinner("載入季度財務數據..."):
        df_quarterly, qtr_error = loader.get_quarterly_data(stock_id)
    
    if qtr_error:
        st.warning(f"⚠️ {qtr_error}")
    elif df_quarterly is not None and not df_quarterly.empty:
        # 金融股：不顯示毛利率（改由提示文字說明可能有小誤差）
        is_finance = bool(df_quarterly.get('是否金融股', False).iloc[-1]) if '是否金融股' in df_quarterly.columns else False
        gp_available = ('毛利率' in df_quarterly.columns) and df_quarterly['毛利率'].notna().any()
        if is_finance and (not gp_available):
            st.subheader("📊 季營收分析")
            st.caption("＊金融股：季營收由月營收加總，數據可能小誤差值")
        else:
            st.subheader("📊 季營收與季毛利率分析")

        # 顯示最新季度摘要
        latest_qtr = df_quarterly.iloc[-1]
        # 顯示最新季度摘要：一般公司顯示毛利率；金融股只顯示營收
        if pd.notna(latest_qtr['營收']):
            if ('毛利率' in df_quarterly.columns) and pd.notna(latest_qtr.get('毛利率')):
                st.markdown(
                    f"💼 最新季度：**{latest_qtr['年度']}Q{latest_qtr['季度']}** | "
                    f"營收 **{latest_qtr['營收']/100000000:.2f}** 億元 | "
                    f"{latest_qtr.get('毛利率名稱','毛利率')} **{latest_qtr['毛利率']:.2f}%**",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"💼 最新季度：**{latest_qtr['年度']}Q{latest_qtr['季度']}** | "
                    f"營收 **{latest_qtr['營收']/100000000:.2f}** 億元",
                    unsafe_allow_html=True
                )
        
        # 繪製季度圖表
        fig_quarterly = plot_quarterly_chart(df_quarterly, stock_id, stock_name)
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            "padding:12px;margin-bottom:10px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>",
            unsafe_allow_html=True
        )
        st.plotly_chart(
            fig_quarterly,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{stock_id}_{stock_name}_quarterly',
                    'height': 600,
                    'width': 1400,
                    'scale': 2
                }
            },
            key=f'quarterly_{stock_id}'
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # AI 分析
    st.markdown("---")
    st.markdown("""
    <h2 style='display: inline-block;'>🤖 AI 深度解盤</h2>
    """, unsafe_allow_html=True)
    
    if api_key:
        with st.spinner("🧠 AI 深度分析中（約 3~4 分鐘）..."):
            # 把月營收/季營收與毛利率摘要塞給AI，讓第四章「財務體質」一定能引用數字
            fundamental_summary = ""
            
            # === 月營收數據 ===
            try:
                df_rev, rev_error = loader.get_monthly_revenue(stock_id)
                if df_rev is not None and not df_rev.empty:
                    rev_tail = df_rev.tail(12).copy()  # 取最近12個月
                    
                    # 處理欄位名稱
                    col_date = '日期' if '日期' in rev_tail.columns else ('date' if 'date' in rev_tail.columns else None)
                    col_rev  = '營收' if '營收' in rev_tail.columns else ('revenue' if 'revenue' in rev_tail.columns else None)
                    col_yoy  = '年增率' if '年增率' in rev_tail.columns else ('yoy' if 'yoy' in rev_tail.columns else None)
                    
                    if col_date and col_rev:
                        # 轉換營收為千元單位
                        rev_tail[col_rev] = (rev_tail[col_rev] / 1000).round(0).astype(int)
                        
                        # 格式化年增率為2位小數
                        if col_yoy:
                            rev_tail[col_yoy] = rev_tail[col_yoy].round(2)
                        
                        # 準備顯示欄位
                        display_cols = [col_date, col_rev]
                        if col_yoy:
                            display_cols.append(col_yoy)
                        
                        fundamental_summary += "【月營收數據（近12個月）】\n"
                        fundamental_summary += "註：營收單位為千元，年增率單位為%\n"
                        fundamental_summary += rev_tail[display_cols].to_string(index=False)
                        fundamental_summary += "\n\n"
            except Exception as e:
                print(f"[WARNING] 月營收數據獲取失敗: {e}")
            
            # === 季營收與毛利率數據 ===
            try:
                df_q, qtr_error = loader.get_quarterly_data(stock_id)
                if df_q is not None and not df_q.empty:
                    q_tail = df_q.tail(8).copy()  # 取最近8季
                    
                    # 轉換營收為千元單位
                    if '營收' in q_tail.columns:
                        q_tail['營收'] = (q_tail['營收'] / 1000).round(0).astype(int)
                    
                    # 格式化毛利率為2位小數
                    if '毛利率' in q_tail.columns:
                        q_tail['毛利率'] = q_tail['毛利率'].round(2)
                    
                    # 準備顯示欄位
                    cols = [c for c in ['季度標籤', '營收', '毛利率'] if c in q_tail.columns]
                    
                    if cols and len(cols) >= 2:  # 至少要有季度標籤和營收
                        fundamental_summary += "【季營收與毛利率數據（近8季）】\n"
                        fundamental_summary += "註：營收單位為千元，毛利率單位為%\n"
                        fundamental_summary += q_tail[cols].to_string(index=False)
                        fundamental_summary += "\n"
            except Exception as e:
                print(f"[WARNING] 季營收數據獲取失敗: {e}")
            
            # 檢查是否有數據
            if not fundamental_summary.strip():
                fundamental_summary = "（暫無月營收/季營收數據可供分析）"
            else:
                print(f"[INFO] 成功準備財務數據，長度: {len(fundamental_summary)} 字元")

            ai_report = analyze_stock_trend(api_key, stock_id, stock_name, df, fundamental_summary=fundamental_summary, holding_summary=holding_summary)
            ai_report = _highlight_ai_report(ai_report)
            st.markdown(f"<div class='ai-report'>{ai_report}</div>", unsafe_allow_html=True)
    else:
        st.warning("請輸入 API Key 以啟用 AI 分析")

else:
    # ========== 起始畫面 ==========

    st.markdown("""
    <div style='background:linear-gradient(135deg,#5b4b9e 0%,#8a4fae 100%);
                border-radius:16px;padding:26px 36px;margin-bottom:22px;
                box-shadow:0 4px 18px rgba(90,70,180,0.18);'>
        <div style='color:#ffffff;font-size:26px;font-weight:900;letter-spacing:0.5px;'>
            📊 台股 AI 分析平台
        </div>
        <div style='color:rgba(255,255,255,0.85);font-size:14px;margin-top:6px;'>
            即時指數動態 · 資金流向 · 個股技術／籌碼／基本面診斷
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;color:#777777;font-size:15px;margin-bottom:24px;'>
        👈 請在左側輸入股票代碼與 API Key 後開始個股分析
    </div>
    """, unsafe_allow_html=True)

    # ── 股市全動態區塊（白色／淺色卡片風格，與美股總經雷達一致）──────────────
    st.markdown("""
    <div style='background: #ffffff;
                border: 1px solid #e3ddf5; border-radius: 16px; padding: 28px 36px;
                margin-bottom: 20px; box-shadow: 0 2px 14px rgba(90,70,180,0.10);'>
        <div style='display:flex; align-items:center; gap:14px; margin-bottom:8px;'>
            <span style='font-size:38px;'>📡</span>
            <div>
                <div style='color:#5b4b9e; font-size:28px; font-weight:900; line-height:1.2;'>
                    股市全動態
                </div>
                <div style='color:#777777; font-size:14px; margin-top:4px;'>
                    AI 即時搜尋 · 大盤指數動態 · 近30日類股資金流向
                </div>
            </div>
        </div>
        <div style='color:#555555; font-size:15px; line-height:1.8; margin-top:10px;
                    border-top:1px solid #ece9f7; padding-top:14px;'>
            🔍 由 Gemini AI 搭配 Google 即時搜尋，先研判加權指數與櫃買指數近
            <strong style='color:#5b4b9e'>30 個交易日</strong>的技術結構（高低位置、動能強弱、是否有反轉徵兆），
            再分析各大類股資金水位的流入流出變化，找出資金輪動路徑。
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 初始化 session state
    if "market_flow_report" not in st.session_state:
        st.session_state["market_flow_report"] = None
    if "tw_index_charts" not in st.session_state:
        st.session_state["tw_index_charts"] = None

    col_btn1, col_btn2, col_spacer = st.columns([2, 2, 6])
    with col_btn1:
        run_market_flow = st.button(
            "📡 啟動股市全動態分析", type="primary",
            use_container_width=True, help="需要輸入 Gemini API Key 才能啟用"
        )
    with col_btn2:
        if st.session_state["market_flow_report"]:
            if st.button("🔄 重新分析", use_container_width=True):
                st.session_state["market_flow_report"] = None
                st.session_state["tw_index_charts"] = None
                st.rerun()

    if run_market_flow:
        if not api_key:
            st.warning("⚠️ 請先在左側輸入 Gemini API Key 才能使用股市全動態功能")
        else:
            with st.spinner("📈 正在抓取大盤指數近30日走勢圖..."):
                st.session_state["tw_index_charts"] = get_tw_index_charts()
            with st.spinner("📡 AI 正在搜尋近30日台股市場動態，分析類股資金流向（約 30~60 秒）..."):
                report = analyze_market_flow(api_key, chart_data=st.session_state["tw_index_charts"])
                st.session_state["market_flow_report"] = report

    # 顯示報告
    if st.session_state["market_flow_report"]:
        report_text = st.session_state["market_flow_report"]

        st.markdown("""
        <div style='background:#f4f1fc;
                    border-left:6px solid #5b4b9e;border-radius:10px;
                    padding:18px 28px;margin-top:20px;margin-bottom:4px;'>
            <span style='color:#5b4b9e;font-size:24px;font-weight:900;'>
                📊 台股股市全動態月報
            </span>
            <span style='color:#888888;font-size:14px;margin-left:16px;'>
                · AI 即時搜尋生成 · 近30個交易日分析
            </span>
        </div>
        """, unsafe_allow_html=True)

        chart_cards_html = _render_index_chart_cards(
            st.session_state["tw_index_charts"],
            title="📈 大盤指數近30日走勢一覽", columns=2
        )
        rendered = _render_market_report(report_text)
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            f"padding:36px 44px;margin-top:4px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>"
            f"{chart_cards_html}{rendered}</div>",
            unsafe_allow_html=True
        )

    # ── 美股總經雷達區塊（白色／淺色卡片風格）────────────────────────────
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #ffffff;
                border: 1px solid #e3ddf5; border-radius: 16px; padding: 28px 36px;
                margin-bottom: 20px; box-shadow: 0 2px 14px rgba(90,70,180,0.10);'>
        <div style='display:flex; align-items:center; gap:14px; margin-bottom:8px;'>
            <span style='font-size:38px;'>🌐</span>
            <div>
                <div style='color:#5b4b9e; font-size:28px; font-weight:900; line-height:1.2;'>
                    美股總經雷達
                </div>
                <div style='color:#777777; font-size:14px; margin-top:4px;'>
                    AI 即時搜尋 · 近30日指數 · 資金板塊流向 · 攻守勢研判
                </div>
            </div>
        </div>
        <div style='color:#555555; font-size:15px; line-height:1.8; margin-top:10px;
                    border-top:1px solid #ece9f7; padding-top:14px;'>
            🔍 由 Gemini AI 搭配 Google 即時搜尋，分析近 <strong style='color:#5b4b9e'>30 日</strong>
            費城半導體、那斯達克、標普500、美元指數、VIX、10年期公債殖利率，
            並解讀 CPI、非農就業、失業率、FED 利率決策等總經數據，研判美股資金板塊流向（科技 vs 傳產/金融）及攻守勢。
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "us_market_report" not in st.session_state:
        st.session_state["us_market_report"] = None
    if "us_index_charts" not in st.session_state:
        st.session_state["us_index_charts"] = None

    col_us1, col_us2, col_us_spacer = st.columns([2, 2, 6])
    with col_us1:
        run_us_market = st.button(
            "🌐 啟動美股總經雷達", type="primary",
            use_container_width=True, key="btn_us_market",
            help="需要輸入 Gemini API Key 才能啟用"
        )
    with col_us2:
        if st.session_state["us_market_report"]:
            if st.button("🔄 重新分析", use_container_width=True, key="btn_us_refresh"):
                st.session_state["us_market_report"] = None
                st.session_state["us_index_charts"] = None
                st.rerun()

    if run_us_market:
        if not api_key:
            st.warning("⚠️ 請先在左側輸入 Gemini API Key 才能使用美股總經雷達功能")
        else:
            with st.spinner("📈 正在抓取六大指數近30日走勢圖..."):
                st.session_state["us_index_charts"] = get_us_index_charts()
            with st.spinner("🌐 AI 正在搜尋近30日美股總經數據，分析資金板塊流向（約 30~60 秒）..."):
                us_report = analyze_us_market(api_key, chart_data=st.session_state["us_index_charts"])
                st.session_state["us_market_report"] = us_report

    if st.session_state["us_market_report"]:
        us_report_text = st.session_state["us_market_report"]

        st.markdown("""
        <div style='background:#f4f1fc;
                    border-left:6px solid #5b4b9e;border-radius:10px;
                    padding:18px 28px;margin-top:20px;margin-bottom:4px;'>
            <span style='color:#5b4b9e;font-size:24px;font-weight:900;'>
                🌐 美股總經雷達月報
            </span>
            <span style='color:#888888;font-size:14px;margin-left:16px;'>
                · AI 即時搜尋生成 · 近30日總經分析
            </span>
        </div>
        """, unsafe_allow_html=True)

        chart_cards_html = _render_us_index_chart_cards(st.session_state["us_index_charts"])
        rendered_us = _render_us_market_report(us_report_text)
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #ece9f7;border-radius:12px;"
            f"padding:36px 44px;margin-top:4px;box-shadow:0 1px 8px rgba(90,70,180,0.06)'>"
            f"{chart_cards_html}{rendered_us}</div>",
            unsafe_allow_html=True
        )