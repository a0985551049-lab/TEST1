import requests
import json
import datetime
import time
import re
import pandas as pd

def fetch_news_summary(api_key, stock_id, stock_name):
    """使用 gemini-2.5-flash + Google Search Grounding 抓取最新新聞摘要"""
    try:
        # v1beta 支援 google_search grounding 工具
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": (
                        f"請用繁體中文搜尋並摘要「台股 {stock_id} {stock_name}」最近的重要新聞，"
                        f"包含：最新財報、法人評等、重大訊息、產業動態、題材催化劑。"
                        f"請條列最多 8 則，每則 2~3 句，並標明來源與日期（若有）。"
                        f"若無相關新聞，回覆「查無近期新聞」。"
                    )
                }]
            }],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if "text" in p]
                news_text = "\n".join(texts).strip()
                if news_text:
                    return news_text
        return ""
    except Exception:
        return ""


def analyze_stock_trend(api_key, stock_id, stock_name, df, fundamental_summary=None):
    """AI 深度分析 - 動態年份版本"""
    
    if not api_key: 
        return "⚠️ 請先輸入 API Key"
    
    try:
        # 數據整理
        essential_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA20', 'MA100', '外資', '投信', '融資餘額']
        valid_cols = [c for c in essential_cols if c in df.columns]
        recent_df = df[valid_cols].tail(30).copy()  # 改為30日
        
        # ✅ 完整的 K 線型態判讀邏輯（參考 quantpass 技術分析標準）
        def classify_kbar(row):
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            body = abs(c - o)
            total_range = h - l
            
            # 防止除以零
            if total_range < 0.001:
                return '一字線'
            
            # 計算上下影線長度
            if c >= o:  # 紅K
                upper_shadow = h - c
                lower_shadow = o - l
            else:  # 黑K
                upper_shadow = h - o
                lower_shadow = c - l
            
            body_ratio = body / total_range if total_range > 0 else 0
            chg_pct = abs(c - o) / o * 100 if o > 0 else 0  # 單日漲跌幅%
            
            # === 1. 十字線系列（開盤價≈收盤價） ===
            if body_ratio < 0.05:  # 實體極小，開盤≈收盤
                # (1) 一字線：開盤=最高=最低=收盤
                if total_range / o < 0.003:
                    return '一字線'
                # (2) T字線：開盤=最高=收盤，有長下影線
                elif upper_shadow < total_range * 0.1 and lower_shadow > body * 2:
                    return 'T字線'
                # (3) 倒T線：開盤=最低=收盤，有長上影線
                elif lower_shadow < total_range * 0.1 and upper_shadow > body * 2:
                    return '倒T線'
                # (4) 標準十字線：有明顯上下影線
                else:
                    return '十字線'
            
            # === 2. 實體K線（影線佔比20%以內） ===
            shadow_ratio = (upper_shadow + lower_shadow) / total_range
            
            if shadow_ratio <= 0.2:
                if c > o:  # 紅K
                    if body_ratio > 0.7 and chg_pct >= 7:
                        return '大紅K'
                    elif body_ratio > 0.4 and chg_pct >= 3:
                        return '中紅K'
                    else:
                        return '小紅K'
                else:  # 黑K
                    if body_ratio > 0.7 and chg_pct >= 7:
                        return '大黑K'
                    elif body_ratio > 0.4 and chg_pct >= 3:
                        return '中黑K'
                    else:
                        return '小黑K'
            
            # === 3. K線帶上影線（墓碑線系列） ===
            # 特徵：上影線長度 > 實體2倍，無下影線或下影線極短
            elif upper_shadow > body * 2 and lower_shadow < body * 0.3:
                if c >= o:
                    return '倒鎚紅K(墓碑線-上漲)'
                else:
                    return '倒鎚黑K(墓碑線-下跌)'
            
            # === 4. K線帶下影線（吊人線系列） ===
            # 特徵：下影線長度 > 實體2倍，無上影線或上影線極短
            elif lower_shadow > body * 2 and upper_shadow < body * 0.3:
                if c >= o:
                    return '紅K鎚子(吊人線-上漲)'
                else:
                    return '黑K鎚子(吊人線-下跌)'
            
            # === 5. K線帶上下影線（紡錘線系列） ===
            # 特徵：同時有明顯上下影線
            else:
                if c >= o:
                    return '紡錘紅K'
                else:
                    return '紡錘黑K'
        
        recent_df['K線'] = recent_df.apply(classify_kbar, axis=1)
        
        # ✅ 價格/均線：小數點後2位；張數（成交量/法人/融資融券）：整數
        int_cols = {'volume','外資','投信','自營商','主力合計','融資餘額','融券餘額'}
        for col in recent_df.columns:
            if col == 'date' or col == 'K線':
                continue
            if col in int_cols:
                recent_df[col] = pd.to_numeric(recent_df[col], errors='coerce').fillna(0).round(0).astype(int)
            else:
                recent_df[col] = pd.to_numeric(recent_df[col], errors='coerce').round(2)
        recent_data = recent_df.to_string(index=False)
        
        # 動態取得年份
        current_year = datetime.datetime.now().year
        last_year = current_year - 1

        # ⚠️ 下面 prompt 已植入趨勢定義與嚴格規定，其餘格式逐字保留原稿
        prompt = f"""
你是股神等級的「台股首席參謀長」，負責在「AI 股市戰情室」中，針對「{stock_id} {stock_name}」進行極為嚴謹的技術、籌碼與基本面診斷。

**【重要約束與定義】**
1. 在第二章均線分析中，僅能分析 MA20 與 MA100，絕對不可提及 MA5、MA10、MA60、MA120、MA240 等其他均線
2. **均線週期正確定義**：
   - MA20（月線）= 短期趨勢線
   - MA100（百日線）= 中期趨勢線
3. **時間表達方式**：
   - 禁止寫死任何年份（例如「2025年」、「2026年」）
   - 使用「最新資訊」、「近期」、「當前」等動態描述
   - 範例：「根據最新財報」而非「根據2025年財報」
4. **表達方式**：直接描述分析結果，不要在正文中重複列出「最近三個交易日 (20XX-XX-XX...)」等日期羅列
5. **數字格式嚴格規定**：所有數字請務必使用「阿拉伯數字」，絕對不要使用國字數字（例如請寫 150，絕對不可寫一百五十）。
6. **人稱與風格規定**：文章中絕對禁止提到「你」。內容需帶有獨特性財經觀點，延伸前因後果，並讓讀者有被激勵感與共感。
7. **數據呈現規定**：須說明內文的重點數據，且數據應自然融入段落文字中，絕對禁止使用條列式列出數據。

**嚴格要求：以下五大章節必須全部完整輸出，每個章節都要有充足內容，絕對不可以中途停止！**

---

### **第一章：K線型態精密掃描** (至少 200 字)
分析最近 1-3 日的 K 棒組合型態與市場情緒變化：

**重要**：數據中的「K線」欄位已標示型態（如大紅K、十字線、倒鎚紅K等），直接引用型態名稱描述近期演變即可，
**絕對不要在報告內文中逐條解釋每種型態的定義、判讀邏輯、實體佔比或影線佔比等理論教學內容**，
讀者只需要看懂「目前是什麼狀況、代表什麼意義、後續怎麼看」，不需要被教育型態學知識。

**分析要點（精簡呈現，避免長篇理論）：**

1. **K棒演變描述**：用「→」符號串連近期 K 線型態的演變，並用「」框起，例如：
   「大紅K強勢上攻 → 倒鎚紅K追高遇壓 → 紡錘黑K多空交戰」
   說明完演變後，直接給一句話結論（例如：短線動能轉弱、留意獲利了結賣壓），不需逐一解釋每個型態代表什麼。

2. **信心評分**：型態可靠度評分（1-5分），一句話說明理由即可，不需長篇分析。

3. **操作思路**：一句話給出「若欲操作」或「積極者可考慮」的參考方向。

---

### **第二章：均線與趨勢結構** (至少 200 字)
**僅分析 MA20 與 MA100，請務必從提供的數據中讀取這兩條均線的數值，絕對禁止提及 MA5、MA60 等其他均線**

請直接給結論，不需要教學均線的定義或計算方式：

* MA20（短期趨勢）與 MA100（中期趨勢）目前的相對位置，判斷屬於多頭排列、空頭排列、多箱或空箱（定義見上方約束）
* 股價與 MA20、MA100 的乖離率是否偏離過大（超買/超賣）
* 目前趨勢的強度與是否有轉折跡象
* 關鍵支撐/壓力價位（MA20、MA100 位置）

---

### **第三章：大戶籌碼與散戶動向** (至少 400 字)
**請分析近 30 個交易日（約一個月）的籌碼變化**

* **外資動向**：
  - 近 30 日累計買賣超張數與趨勢
  - 操作態度解讀（持續加碼/減碼/觀望）
  
* **投信籌碼**：
  - 近 30 日買賣超統計
  - 持股變化與操作態度
  
* **融資融券**：
  - 融資餘額增減意義
  - 散戶情緒判斷
  
* **籌碼總結**：
  - 主力集中度評估（法人買 vs 散戶賣，或相反）
  - 籌碼安定性與壓力

---

### **第四章：產業與基本面展望** (至少 800 字)
* **公司定位**：
  - 主要產品服務與產業鏈位置
  - 核心競爭優勢
  - 主要客戶與市場
  
* **產業趨勢**：
  - 當前產業景氣狀況（使用「最新趨勢」而非「{last_year}-{current_year}年趨勢」）
  - 成長動能與挑戰
  
* **題材催化劑**：
  - 當前熱門題材（AI、半導體等）
  - 正面/負面因素
  
* **財務體質**：
  - 最新營收獲利表現（使用「最新財報」而非具體年份）
  - 毛利率、淨利率(如果是金融股，則不分析這兩項)
  - 財務穩健度
  
* **法人觀點**：
  - 券商目標價
  - 市場共識

---

### **第五章：最終操作策略** (至少 500 字)
* **多空方向**：
  - 明確表態與操作時間軸
  - 綜合評分依據
  
* **關鍵價位**：
  - 支撐位：第一、第二支撐（MA20 為短期支撐，MA100 為中期支撐）
  - 壓力位：第一、第二壓力
  - 止損價位
  
* **積極型建議**：
  - 進場時機與價位
  - 停損設定
  - 獲利目標
  
* **保守型建議**：
  - 觀察訊號
  - 防守策略
  
* **風險提示**：
  - 情境預測
  - 風險因子
(1)請使用條列式，每個風險獨立編號，(2)移除所有雙星號(**)

**【重要聲明】**
- 使用「若欲操作」、「可考慮」、「參考思路」等詞彙
- 避免使用「建議」、「應該」、「必須」等指示性用語
- 強調這是「技術分析參考」而非「投資建議」

---

### **診斷結語：優劣勢總表與入場參考** (必須完整輸出)

請在第五章結束後，輸出以下格式的診斷結語（純文字 Markdown 表格格式，不要加 HTML 標籤）：

**格式要求如下：**

1. 輸出一個「優勢 vs 劣勢」對照表格，格式如下：
```
| 面向 | 優勢 ✅ | 劣勢 ❌ |
|------|---------|---------|
| 技術面 | （技術優勢描述，15字內） | （技術劣勢描述，15字內） |
| 籌碼面 | （籌碼優勢描述，15字內） | （籌碼劣勢描述，15字內） |
| 基本面 | （基本面優勢描述，15字內） | （基本面劣勢描述，15字內） |
| 產業面 | （產業優勢描述，15字內） | （產業劣勢描述，15字內） |
| 風險面 | （風險優勢描述，15字內） | （風險劣勢描述，15字內） |
```

2. 接著輸出「入場參考價位」表格，格式如下：
```
| 類型 | 價位（元） | 說明 |
|------|-----------|------|
| 積極入場參考 | （價格） | （簡短說明，20字內） |
| 保守入場參考 | （價格） | （簡短說明，20字內） |
| 短期目標參考 | （價格） | （簡短說明，20字內） |
| 中期目標參考 | （價格） | （簡短說明，20字內） |
| 止損參考 | （價格） | （簡短說明，20字內） |
```

3. 重要規範：
   - 所有描述使用繁體中文
   - 價格數字使用阿拉伯數字，具體明確
   - 表格必須是標準 Markdown 格式（以 | 開頭結尾）
   - 入場/目標/止損均使用「參考」字樣，避免指示性用語
   - 此部分標題為「### 診斷結語：優劣勢總表與入場參考」

---

**近 30 日完整數據（包含 MA20 與 MA100）**
{recent_data}

**【重要】月營收與季營收數據（第四章財務體質必須使用）**
{fundamental_summary if fundamental_summary else "（暫無月營收/季營收數據）"}

**輸出規則**
1. 繁體中文，Markdown 格式
2. 語氣專業犀利
3. 每章節必須完整
4. 總字數 2200+ 字
5. 數據具體明確
6. 禁止寫死任何年份數字

7. **【嚴格要求】第四章財務體質部分：**
   - 如果上方有提供月營收/季營收數據，你**必須**直接引用這些具體數字進行分析
   - 不可以說「缺乏數據」或「無法獲得數據」
   - 必須分析營收趨勢、年增率變化、毛利率走勢等具體數值
   - 例如：「最近一個月營收為 150 億元，年增率為 +/-10%」

8. **【重要】用詞規範（避免法律風險）- 絕對不可使用投資指示用語：**
   - ❌ 絕對禁用：「建議」、「應該」、「必須」、「強烈推薦」、「推薦」、「買入」、「賣出」、「進場」、「出場」、「加碼」、「減碼」
   - ✅ 改用：「若欲操作」、「可考慮」、「積極者可留意」、「參考思路」、「值得觀察」、「可能」、「或許」
   - 範例：「若欲操作，可考慮在 XX 元附近觀察」而非「建議在 XX 元買進」
   - 範例：「停損可參考設定在 XX 元」而非「應該將停損設在 XX 元」
   - 範例：「積極者可留意 XX 元附近的機會」而非「推薦在 XX 元進場」
   - 所有操作相關內容都要強調「僅供參考」、「學術研究」性質
   - 整篇文章請全面檢查，確保沒有任何投資指示用語

9. **【格式要求】第一章 K 線型態描述：**
   - 必須使用「→」符號串連演變過程
   - 用「」框起整個演變描述
   - 範例：「小陽線觀望 → 大陽線帶長上影線追價遇壓 → 實體極小帶長下影線多空平衡」

10. **【格式要求】移除所有雙星號（**）：**
   - 副標題不使用 ** 包圍，直接呈現文字
   - 例如：「月營收分析」而非「**月營收分析**」
   - 例如：「技術面」而非「**技術面**」
   - 整篇文章不使用 ** 來強調，改用具體描述

11. **【數字格式化要求】- 非常重要：**
   - 所有百分比（年增率、毛利率等）：僅保留小數點後2位（例如：-36.61%，不是-36.612984%）
   - 營收數據已換算為千元單位，請直接使用並標註「千元」（例如：營收 165,191 千元）
   - 不要將營收寫成「1,855,499,000 元」，要寫成「1,855,499 千元」
   - 確保所有數字格式統一、易讀
"""
        
        # ========== 抓取最新新聞（Google Search Grounding）==========
        news_summary = fetch_news_summary(api_key, stock_id, stock_name)
        if news_summary:
            prompt += f"""

---

**【最新新聞摘要（Google 即時搜尋）】**
{news_summary}

**重要指示**：上方新聞為即時搜尋結果，請在第四章「產業與基本面展望」中適當引用這些最新資訊，包含最新財報、法人評等、產業動態等，讓分析更具時效性與參考價值。
"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": { 
                "temperature": 0.4,
                "maxOutputTokens": 16384,
                "topP": 0.95,
                "topK": 40
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        # 優先使用穩定、配額較寬鬆的模型
        model_attempts = [
            "gemini-3-pro-preview",           # 最新版 Pro（優先，需付費）
            "gemini-3-flash-preview",         # 最新版 Flash（次優先，有免費額度）
            "gemini-2.0-flash-exp",           # 實驗版，通常配額較多
            "gemini-1.5-flash-latest",        # 穩定版 Flash
            "gemini-1.5-pro-latest",          # 穩定版 Pro
            "gemini-2.5-flash"                # 新版 Flash
        ]

        last_error = None
        
        for idx, model_name in enumerate(model_attempts):
            # 非第一個模型時，等待 3 秒避免速率限制
            if idx > 0:
                time.sleep(3)  # 移除 print，靜默等待
            try:
                # gemini-3 preview 需走 v1beta；其餘走 v1
                api_ver = "v1beta" if model_name.startswith("gemini-3") else "v1"
                url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"

                # ✅ 每次請求前稍微延遲，避免觸發速率限制
                model_success = False
                for attempt in range(3):
                    if attempt > 0:
                        # 重試時等待更久
                        delay = min(15, 3 * (2 ** attempt))
                        time.sleep(delay)  # 移除 print，靜默等待
                    
                    response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=90)

                    if response.status_code == 200:
                        result = response.json()
                        if 'candidates' in result and len(result['candidates']) > 0:
                            text = result['candidates'][0]['content']['parts'][0]['text']
                            return f"### 🧬 AI 戰情室：全方位深度解析\n\n{text}\n\n---\n**使用模型**: {model_name}"
                        last_error = f"{model_name} HTTP 200 但回傳格式異常: {str(result)[:300]}"
                        break

                    if response.status_code == 429:
                        try:
                            err = response.json()
                            msg = err.get("error", {}).get("message", "")
                            m = re.search(r"Please retry in ([0-9.]+)s", msg)
                            # 限制最長等待時間為 10 秒，避免等太久
                            wait_s = min(10, float(m.group(1)) if m else (2 ** attempt) * 2)
                        except Exception:
                            wait_s = min(10, (2 ** attempt) * 2)

                        last_error = f"{model_name} HTTP 429 (attempt {attempt+1}/3): quota/rate limit"
                        time.sleep(wait_s)  # 移除 print，靜默等待
                        continue  # 繼續下一次重試

                    # 其他 HTTP 錯誤（400, 404, 500 等）直接跳過這個模型
                    last_error = f"{model_name} HTTP {response.status_code}: {response.text[:800]}"
                    break  # 跳出重試迴圈，嘗試下一個模型

                # 如果這個模型的所有重試都失敗了，繼續嘗試下一個模型
                if not model_success:
                    continue  # 移除 print，靜默切換到下一個模型

            except Exception as e:
                last_error = f"{model_name} Exception: {str(e)}"
                continue  # 移除 print，靜默嘗試下一個模型
                
        return f"❌ 所有模型皆無法連線，請檢查 API Key / 額度 / 網路狀態\n\n最後錯誤：{last_error}"

    except Exception as e:
        return f"系統錯誤: {str(e)}"

def analyze_market_flow(api_key):
    """
    使用 Gemini + Google Search Grounding 分析台股大盤動態與近14日資金流向
    結構比照「美股總經雷達」：先看大盤指數的技術結構（高低位置、動能、反轉徵兆），
    再看類股資金水位變化，多模型輪詢，自動 fallback，表格格式嚴格規範。
    """
    if not api_key:
        return "⚠️ 請先輸入 API Key 以啟用股市全動態分析"

    try:
        import datetime
        today = datetime.date.today().strftime("%Y年%m月%d日")

        prompt = f"""
今天是 {today}。請搜尋最新台股市場資訊，並用繁體中文撰寫一份「台股股市全動態雙週報」。

---

### 第一節：大盤指數近14日動態研判（至少400字）

**重要方法論：判斷大盤多空與強弱時，禁止只看「近14日漲跌幅為正／負」就直接貼上「強勢／多頭」或「弱勢／空頭」標籤。
解讀文字中「必須」包含以下三個要素，缺一不可：**
1. 目前指數相對於近14日高低點的相對位置（貼近高點、貼近低點、還是區間中段，並寫出概略距離）
2. 動能強弱的具體徵兆：若貼近或創近14日新高，需說明「爬升動能是否減弱」（例如：量縮上攻、漲勢趨緩、
   連續無法有效站穩新高、上影線增多、成交量能是否同步放大等現象）；若貼近或創近14日新低，需說明是
   「加速趕底」還是「跌深後出現止穩訊號」。不可以只寫「強勢」「弱勢」「多頭」「空頭」這類單一詞彙就結束。
3. 綜合研判用語只能從以下語彙中選用最貼切者（可微調文字但語意需相同）：
   「續強格局」、「衝高乏力／留意拉回風險」、「創高但動能背離／反轉疑慮升高」、
   「續弱格局」、「跌深有撐／醞釀反彈」、「破底加速趕底」、「區間整理／方向未明」

請依此方法論搜尋並說明：

1. **加權指數（TAIEX）**：近14日高低點、目前位置、爬升或下跌動能是否減弱、成交量能變化
2. **櫃買指數（OTC）**：近14日走勢、目前位置、與加權指數的背離或共振
3. **法人期貨籌碼氛圍**：外資台指期未平倉、選擇權籌碼所反映的多空氣氛（若可搜尋到相關數據）

接著撰寫【第一節綜合結論】（至少100字），說明大盤目前的技術結構定性，以及是否存在「表面數字上漲但結構轉弱」或「表面下跌但已現止穩」的落差。

---

### 第二節：近14日類股資金流向地圖（至少600字）

**重要方法論：判斷資金流入/流出時，請評估「資金水位的變化趨勢」，而不是只看單日或單期的買賣超金額大小。**
具體來說：
- 請比較近14日每週（或每3~5個交易日為一區間）外資／投信買賣超金額的「累積水位」如何變化，
  是持續墊高（水位向上）、持續探底（水位向下）、還是高檔鈍化／低檔鈍化（水位持平但已達相對極端）。
- 一天的大量買超若只是相對於前一週持續探底的水位小幅反彈，不應直接判定為「資金流入」；
  需說明該水位目前處於近14日的相對高檔、中段還是低檔位置。
- 請具體寫出水位變化的方向與位置，例如：「外資買超金額連續5日墊高，水位已由近14日低點回升至中高檔」
  或「投信買超金額雖單日創高，但仍低於近14日平均水位，尚未脫離弱勢區間」。

請依此方法論分析：
1. **資金流出類股**：哪些類股資金水位明顯向下探底？請列出前3名，每個至少80字，並說明水位變化的具體軌跡（不是只看單日金額）。
2. **資金流入類股**：哪些類股資金水位明顯墊高？請列出前3名，每個至少80字，並說明水位變化的具體軌跡。
3. **仍受關注類股**：哪些類股資金水位持續維持高檔或中性盤整，籌碼穩定？
4. **資金輪動路徑**：用「A類股 → B類股 → C類股」格式描述近14日資金水位輪動路徑

接著請輸出一個「近14日類股流入/流出排名（前5名）」表格，格式如下（每行都必須填入實際數據，嚴格按照此格式）：

| 類股名稱 | 資金水位變化 | 主要驅動力/原因 | 代表個股（近14日表現） |
|------|------|------|------|
| 半導體 | 水位持續墊高 | AI需求強勁，台積電法說樂觀 | 台積電+8% |
| 金融股 | 水位持續探底 | 升息預期降溫，外資調節 | 富邦金-3% |

**重要：表格必須包含至少5行資料，每行都要有實際內容，不可留空，也不可以輸出整排的「-」或空白佔位符號。**

---

### 第三節：近兩週市場總結與下週展望（至少300字）

- 近14日大盤走勢摘要（高點、低點、區間，呼應第一節的技術結構判斷）
- 外資、投信近14日買賣超「水位變化趨勢」概況（不是單日金額，而是連續變化的方向與相對位置）
- 下週關鍵事件
- 風險提示

---

**規範：** 阿拉伯數字、避免投資指示用語、僅供學術研究、總字數1500字以上、只需輸出第一、二、三節，不要輸出個股推薦內容
"""

        report, err = _gemini_generate(api_key, prompt, max_tokens=8192)
        if report:
            return _strip_grounding_citations(report)
        return f"❌ 所有模型皆無法取得市場數據，請稍後再試。\n\n最後錯誤：{err}"

    except Exception as e:
        return f"❌ 系統錯誤：{str(e)}"


def _strip_grounding_citations(text: str) -> str:
    """
    清除 Gemini Google Search grounding 殘留的引用標記，例如：
    [cite: 37 (previous search)]、[cite: 9, 23]、[cite_start]...[cite_end] 等，
    這些是模型內部的搜尋引用註記，不應該顯示給使用者。
    """
    if not isinstance(text, str):
        return text
    # 涵蓋 [cite: ...]、[cite_start]、[cite_end] 等變體，中括號內含 cite 字樣者一律移除
    text = re.sub(r'\[\s*cite[^\]]*\]', '', text, flags=re.IGNORECASE)
    # 清除移除後可能留下的多餘空白（例如句尾多一個空格再接句號）
    text = re.sub(r'[ \t]+([。，、；：])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


def _gemini_generate(api_key, prompt, max_tokens=8192, temperature=0.3):
    """
    共用的 Gemini 呼叫邏輯：多模型輪詢 + 自動重試 + Google Search grounding。
    回傳 (report_text, error_message)，成功時 error_message 為 None。
    """
    if not api_key:
        return None, "⚠️ 尚未提供 API Key"

    headers = {"Content-Type": "application/json"}
    model_attempts = [
        ("v1beta", "gemini-2.5-flash",               True),
        ("v1beta", "gemini-2.5-flash-preview-05-20", True),
        ("v1beta", "gemini-2.0-flash-exp",           True),
        ("v1",     "gemini-1.5-flash-latest",        False),
        ("v1",     "gemini-1.5-pro-latest",          False),
        ("v1beta", "gemini-1.5-flash",               True),
    ]

    last_error = None
    for api_ver, model_name, use_grounding in model_attempts:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }
        if use_grounding:
            payload["tools"] = [{"google_search": {}}]

        for attempt in range(3):
            if attempt > 0:
                time.sleep(min(10, 3 * (2 ** attempt)))
            try:
                response = requests.post(
                    f"{url}?key={api_key}", headers=headers,
                    json=payload, timeout=120
                )
            except Exception as e:
                last_error = f"{model_name} 連線例外: {str(e)}"
                break

            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates:
                    finish_reason = candidates[0].get("finishReason", "")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    texts = [p.get("text", "") for p in parts if "text" in p]
                    report = "\n".join(texts).strip()
                    if report:
                        # 若因達到 token 上限被截斷且內容過短，視為失敗改用下一個模型
                        if finish_reason == "MAX_TOKENS" and len(report) < 200:
                            last_error = f"{model_name} 內容被截斷(MAX_TOKENS)且過短"
                            break
                        return _strip_grounding_citations(report), None
                last_error = f"{model_name} HTTP 200 格式異常"
                break

            if response.status_code in (429, 503, 529):
                try:
                    msg = response.json().get("error", {}).get("message", "")
                    m = re.search(r"retry in ([0-9.]+)s", msg)
                    wait_s = min(15, float(m.group(1)) if m else (3 * (2 ** attempt)))
                except Exception:
                    wait_s = min(15, 3 * (2 ** attempt))
                last_error = f"{model_name} HTTP {response.status_code} (attempt {attempt+1}/3)"
                time.sleep(wait_s)
                continue

            last_error = f"{model_name} HTTP {response.status_code}: {response.text[:200]}"
            break

    return None, last_error


def _build_index_session():
    """建立帶瀏覽器 User-Agent 的 requests session，降低 Yahoo Finance 封鎖雲端請求的機率。"""
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        })
        return session
    except Exception:
        return None


def _fetch_yf_closes(yf, symbols, session):
    """依序嘗試每個備援代碼；每個代碼先試 history()，失敗再試 download()。"""
    last_err = None
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol, session=session).history(period="3mo", interval="1d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if len(closes) >= 2:
                    return closes, None
            last_err = f"{symbol}: history() 回傳空資料"
        except Exception as e:
            last_err = f"{symbol}: history() 例外 - {str(e)}"

        try:
            kwargs = {"period": "3mo", "interval": "1d", "progress": False,
                      "threads": False, "auto_adjust": True}
            if session is not None:
                kwargs["session"] = session
            hist2 = yf.download(symbol, **kwargs)
            if hist2 is not None and not hist2.empty and "Close" in hist2.columns:
                closes2 = hist2["Close"].dropna()
                if isinstance(closes2, pd.DataFrame):
                    closes2 = closes2.iloc[:, 0]
                if len(closes2) >= 2:
                    return closes2, None
            last_err = f"{symbol}: history()/download() 皆無資料（{last_err}）"
        except Exception as e2:
            last_err = f"{symbol}: download() 也失敗 - {str(e2)}（前次錯誤：{last_err}）"

    return None, last_err or "無法取得資料"


def _generate_index_chart_items(indices, tail_days=30):
    """
    共用的指數走勢小圖產生邏輯，供美股／台股共用。
    indices: [(符號備援清單, 中文名稱, 代碼, 顯示單位, 數值縮放係數), ...]
    回傳每個指數的最新值、漲跌幅與 base64 PNG 圖檔；任一指數失敗不影響其他指數。
    若 yfinance/matplotlib 未安裝則回傳 None。
    """
    import io
    import base64

    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    session = _build_index_session()

    results = []
    for symbols, name_zh, code, unit, scale in indices:
        item = {
            "symbol": symbols[0], "name": name_zh, "code": code, "unit": unit,
            "latest": None, "change": None, "change_pct": None,
            "img_base64": None, "error": None,
        }
        try:
            closes, err = _fetch_yf_closes(yf, symbols, session)
            if closes is None:
                item["error"] = err or "無法取得資料"
                results.append(item)
                continue

            closes = closes.tail(tail_days) * scale
            if len(closes) < 2:
                item["error"] = "資料不足"
                results.append(item)
                continue

            latest = float(closes.iloc[-1])
            first = float(closes.iloc[0])
            change = latest - first
            change_pct = (change / first * 100.0) if first else 0.0
            is_up = change >= 0

            line_color = "#e63946" if is_up else "#2a9d3e"   # 紅漲綠跌（台股慣例）
            fill_color = "#fde3e3" if is_up else "#dff3e0"

            fig, ax = plt.subplots(figsize=(3.0, 0.9), dpi=140)
            x = range(len(closes))
            ax.plot(x, closes.values, color=line_color, linewidth=1.8)
            ax.fill_between(x, closes.values, closes.values.min(), color=fill_color, alpha=0.7)
            ax.axis("off")
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            plt.tight_layout(pad=0.1)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")

            item.update({
                "latest": latest,
                "change": change,
                "change_pct": change_pct,
                "img_base64": img_b64,
            })
        except Exception as e:
            item["error"] = str(e)
        results.append(item)

    return results


def get_us_index_charts():
    """
    抓取六大美股關鍵指數近30個交易日資料，繪製走勢小圖（sparkline），
    回傳每個指數的最新值、漲跌幅與 base64 PNG 圖檔，供前端直接嵌入 <img>。
    任一指數失敗不影響其他指數；若 yfinance/matplotlib 未安裝則回傳 None。
    """
    # 每項指標可提供多個備援代碼；中文名稱、代碼、顯示單位、數值縮放係數
    # （^TNX 在 Yahoo 上是殖利率*10，需除以10還原）
    indices = [
        (["^SOX"],               "費城半導體指數",   "SOX",     "",  1.0),
        (["^IXIC"],              "那斯達克指數",     "NASDAQ",  "",  1.0),
        (["^GSPC"],              "標普500指數",      "S&P 500", "",  1.0),
        (["DX-Y.NYB", "DX=F"],   "美元指數",         "DXY",     "",  1.0),
        (["^VIX"],               "VIX恐慌指數",      "VIX",     "",  1.0),
        (["^TNX"],               "10年期公債殖利率", "US10Y",   "%", 0.1),
    ]
    return _generate_index_chart_items(indices, tail_days=30)


def get_tw_index_charts():
    """
    抓取台股加權指數（及櫃買指數）近14個交易日走勢，繪製 sparkline 小圖，
    呈現方式與 get_us_index_charts() 一致，供「股市全動態」白底版面使用。
    """
    indices = [
        (["^TWII"],  "台股加權指數", "TAIEX", "",  1.0),
        (["^TWOII"], "櫃買指數",     "OTC",   "",  1.0),
    ]
    return _generate_index_chart_items(indices, tail_days=14)


def analyze_us_market(api_key):
    """
    使用 Gemini + Google Search Grounding 分析近30日美股總體經濟與資金板塊流向
    涵蓋：費城半導體、那斯達克、標普500、美元指數、VIX、
          10年期公債殖利率、CPI、非農就業、失業率、FED利率

    為避免單次生成內容過長遭截斷，拆成前後兩段（第一~三節 / 第四~六節）
    分兩次呼叫 Gemini 後再合併，任一段失敗也不影響另一段的呈現。
    """
    if not api_key:
        return "⚠️ 請先輸入 API Key 以啟用美股總經雷達"

    try:
        today = datetime.date.today().strftime("%Y年%m月%d日")

        prompt_part1 = f"""
今天是 {today}。請搜尋最新美股與總體經濟資訊，用繁體中文撰寫「美股總經雷達月報」的前半部。
請只輸出第一、二、三節，不要輸出其他章節、不要有開場白或結語，直接從「### 第一節」開始寫起。

---

### 第一節：近30日美股關鍵指數快覽（至少600字）

**重要方法論：判斷多空與強弱時，禁止只看「近30日漲跌幅為正／負」就直接貼上「強勢／多頭」或「弱勢／空頭」標籤。
每一項指標的解讀文字中，都「必須」包含以下三個要素，缺一不可：**
1. 目前價位相對於近30日高低點的相對位置（貼近高點、貼近低點、還是區間中段，並寫出概略距離）
2. 動能強弱的具體徵兆：若貼近或創近30日新高，需說明「爬升動能是否減弱」（例如：量縮上攻、漲勢趨緩、
   連續無法有效站穩新高、上影線增多等現象）；若貼近或創近30日新低，需說明是「加速趕底」還是
   「跌深後出現止穩訊號」。不可以只寫「強勢」「弱勢」「多頭」「空頭」這類單一詞彙就結束該指標的分析，
   後面一定要接上「原因」與「後續可能性」的說明。
3. 根據以上兩點的「綜合研判用語」，只能從以下語彙中選用最貼切者（可微調文字但語意需相同），
   不可簡化為單純的多空標籤：
   「續強格局」、「衝高乏力／留意拉回風險」、「創高但動能背離／反轉疑慮升高」、
   「續弱格局」、「跌深有撐／醞釀反彈」、「破底加速趕底」、「區間整理／方向未明」

請搜尋並依序說明以下各指標近30日的走勢數據與市場意義（需包含上述技術結構判斷）：

1. **費城半導體指數（SOX）**：近30日高低點、目前位置、爬升或下跌動能是否減弱、與AI半導體股的關聯
2. **那斯達克指數（NASDAQ）**：近30日走勢、目前位置、動能強弱、與費半的背離或共振
3. **標準普爾500指數（S&P 500）**：近30日走勢、目前位置、動能強弱、大盤健康度、資金廣度
4. **美元指數（DXY）**：近30日走勢、目前位置、對新興市場與台幣的影響
5. **VIX恐慌指數**：目前水位、近30日相對高低位置、市場情緒是「貪婪」還是「恐慌」（15以下=樂觀，15-25=中性，25-35=警戒，35+=極端恐慌）
6. **美國10年期公債殖利率**：目前水位、近30日變動趨勢、目前位置、對成長股估值的壓力

請用「指標名稱：數值 → 解讀（含技術結構判斷）」格式逐一呈現，接著撰寫【第一節綜合結論】（至少150字），說明：
- 六大指標相互之間的共振或背離關係
- 目前市場整體方向的一致性研判（是否有「表面數字上漲但結構轉弱」的隱憂）
- 最值得警惕的訊號組合（如：VIX低+公債升息＝潛在壓力累積；或指數創高但爬升動能減弱＝反轉風險升高）

---

### 第二節：近期重大總經事件解讀（至少400字）

請搜尋並說明：

1. **CPI消費者物價指數**：最新公布數據、年增率、核心CPI、市場反應
2. **非農就業人口（NFP）**：最新公布數字、與預期的比較、就業市場強弱
3. **失業率**：最新數字、趨勢
4. **FED利率決策**：目前聯邦基金利率目標區間、最近一次FOMC決議、點陣圖暗示的降息/升息路徑、市場對下次決議的預期（用CME FedWatch工具數據）

接著撰寫【第二節綜合結論】（至少100字），說明這四大總經數據如何共同形塑當前的「緊縮/寬鬆」政策預期，以及對股市的影響方向。

---

### 第三節：市場熱絡度與恐慌程度綜合評分

**重要方法論：「訊號方向」欄位「禁止」只填寫「強勢」「多頭」「弱勢」「空頭」這類單一詞彙，
必須填入包含技術結構判斷的完整用語，例如「續強／衝高乏力」「創高但動能減弱」「跌深有撐」「破底加速」等。
「市場解讀」欄位請依技術結構（相對高低點位置、動能強弱、是否有背離或反轉跡象）給出判斷。若指數雖上漲但已
接近前波高點且爬升力道減弱，請具體寫出「衝高乏力／有拉回風險」等判斷；若雖下跌但已出現止穩或量縮，也請
具體寫出「跌深有撐／醞釀反彈」等判斷。**

請輸出以下評分表格，每欄必須填入實際數據：

| 指標 | 近期數值 | 訊號方向 | 市場解讀 |
|------|------|------|------|
| VIX恐慌指數 | XX.X | 樂觀/中性/警戒/恐慌 | 說明目前相對高低位置與情緒轉折跡象 |
| 費城半導體30日漲跌 | +X.X% / -X.X% | （依技術結構判斷，非僅看正負號） | 說明是否貼近高點、動能是否減弱 |
| 那斯達克30日漲跌 | +X.X% / -X.X% | （依技術結構判斷，非僅看正負號） | 說明是否貼近高點、動能是否減弱 |
| 10年期公債殖利率 | X.XX% | 升/降/持平 | 說明 |
| 美元指數DXY | XXX.X | 強勢/弱勢 | 說明 |
| 市場總體熱絡度 | （1-10分） | 熱絡/中性/冷清 | 綜合評估 |
| 整體恐慌指數 | （1-10分） | 樂觀/中性/恐慌 | 綜合評估 |

**重要：表格每一列都必須是完整的「| 內容 | 內容 | 內容 | 內容 |」格式，每格都要填入實際數值或文字，絕對不可以輸出整排的「-」、「_」或任何空白佔位符號，也不可以留空或填「N/A」。**

表格後請撰寫【第三節綜合結論】（至少100字），用文字說明熱絡度與恐慌程度評分的整體意義，並指出目前市場處於「風險偏好上升」或「風險規避」模式，以及是否存在「數字看似樂觀但結構已轉弱」的落差。

---

**規範：** 阿拉伯數字、繁體中文、僅供學術研究、只需輸出第一、二、三節（含表格），不要輸出第四節以後的內容，不要有結語。
"""

        prompt_part2 = f"""
今天是 {today}。請搜尋最新美股與總體經濟資訊，用繁體中文撰寫「美股總經雷達月報」的後半部。
請只輸出第四、五、六節，不要輸出第一、二、三節、不要有開場白，直接從「### 第四節」開始寫起。

---

### 第四節：美股資金板塊流向分析（至少500字）

這是本報告最重要的核心分析，請搜尋並說明：

**資金從哪裡流出？**
說明近30日明顯遭到資金撤退的板塊（如：科技成長股、半導體、生技等），原因為何？

**資金流向哪裡？**
說明近30日明顯吸引資金流入的板塊（如：金融、能源、公用事業、傳統製造等），驅動因素？

**攻守勢研判：**
請用「攻勢型」或「守勢型」明確判斷目前市場主流策略，並說明：
- 攻勢型：資金追逐高Beta成長板塊（科技、半導體、AI概念）
- 守勢型：資金轉進防禦板塊（公用事業、金融、消費必需品、REITs）

**輪動路徑描述（必填）：**
請用「A板塊 → B板塊 → C板塊」格式描述近30日美股資金的輪動路徑

輸出「板塊資金流向快覽」表格：

| 板塊名稱 | 資金方向 | 近30日表現 | 主要驅動力 |
|------|------|------|------|
| 科技/AI | 流入/流出 | +X% / -X% | 說明 |
| 半導體 | 流入/流出 | +X% / -X% | 說明 |
| 金融 | 流入/流出 | +X% / -X% | 說明 |
| 能源 | 流入/流出 | +X% / -X% | 說明 |
| 公用事業 | 流入/流出 | +X% / -X% | 說明 |
| 傳統製造/工業 | 流入/流出 | +X% / -X% | 說明 |

**重要：表格每一列都必須是完整的「| 內容 | 內容 | 內容 | 內容 |」格式，每格都要填入實際數值或文字，絕對不可以輸出整排的「-」、「_」或任何空白佔位符號，也不可以留空或填「N/A」。**

---

### 第五節：對台股的影響研判（至少300字）

**重要規範：**
- 本節「禁止」輸出任何表格（不可出現任何 `|` 符號的表格列），只能用文字段落＋條列說明呈現。
- 本節聚焦「傳導到台股」的具體影響，「不要」重複第四節已經寫過的美股板塊資金流向細節數據
  （例如不要再重述一次科技/半導體/金融/能源板塊各自漲跌幾%），只需直接引用結論一句帶過，
  把重心放在下面四個台股專屬的傳導效果上。

請根據以上美股總經分析，用流暢文字（可搭配少量條列）依序說明：
1. 美股趨勢對台股的外資動向影響（外資是否可能加碼/減碼台股）
2. 費半走勢對台積電、聯發科等台灣半導體的牽引力
3. 美元/台幣匯率對外資匯出入的影響
4. 下月需關注的重大美股事件（FOMC、財報季、重要總經數據公布）及其對台股的潛在衝擊

---

### 第六節：本月美股總經全面結論與操作建議（至少300字）

這是本報告的總結，必須完整輸出，不可省略：

**整體市場定性：**
綜合以上五節所有分析，用3-5句話說明目前美股市場的整體健康狀態與主要驅動因素。

**攻守勢最終裁定：**
明確寫出「本月美股整體判斷為：【攻勢型】」或「本月美股整體判斷為：【守勢型】」，並說明判斷依據。

**最大風險警示：**
列出目前最需警惕的1-3個市場風險（如：公債殖利率飆升、VIX突破警戒線、美元急升壓縮新興市場等）。

**對台股投資人的建議方向：**
根據美股板塊流向與總經環境，給出台股投資人應重點關注的方向（如：台積電跟隨費半、金融股受惠降息、防禦型配置等）。僅供學術研究參考。

---

**規範：** 全文使用阿拉伯數字、繁體中文、僅供學術研究、數據務必標明來源日期、每一節都必須完整輸出不可中途停止，只需輸出第四、五、六節。
"""

        report1, err1 = _gemini_generate(api_key, prompt_part1, max_tokens=8192)
        report2, err2 = _gemini_generate(api_key, prompt_part2, max_tokens=8192)

        if not report1 and not report2:
            return f"❌ 所有模型皆無法取得數據，請稍後再試。\n\n最後錯誤：{err1 or ''} {err2 or ''}"

        header = f"## 美股總經雷達月報\n\n**發布日期：{today}**\n\n---\n"

        sections = []
        if report1:
            sections.append(report1)
        else:
            sections.append(f"⚠️ 第一~三節生成失敗，請按「重新分析」再試一次。\n\n（錯誤訊息：{err1}）")

        if report2:
            sections.append(report2)
        else:
            sections.append(f"⚠️ 第四~六節生成失敗，請按「重新分析」再試一次。\n\n（錯誤訊息：{err2}）")

        return header + "\n\n---\n\n".join(sections)

    except Exception as e:
        return f"❌ 系統錯誤：{str(e)}"


def generate_quick_summary(df, name):
    try:
        latest = df.iloc[-1]
        change = latest['close'] - df.iloc[-2]['close']
        pct = (change / df.iloc[-2]['close']) * 100
        color = "🔴" if change > 0 else "🟢" if change < 0 else "⚪"
        return f"{color} {name} 收盤：{latest['close']} ({change:+.2f} / {pct:+.2f}%) | 量 {int(latest['volume'])} 張"
    except:
        return "數據載入中..."