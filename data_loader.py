import pandas as pd
import datetime
from FinMind.data import DataLoader
import streamlit as st
from stock_names import get_stock_name


def _create_finmind_loader():
    """
    建立 FinMind DataLoader，若使用者有在 Secrets 設定 FINMIND_TOKEN 則自動登入，
    免費額度為 300 次/小時，註冊並使用 token 後可提高到 600 次/小時。
    未設定 token 也能正常運作（走匿名額度），不會因此中斷程式。
    """
    dl = DataLoader()
    try:
        token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        token = ""
    if token:
        try:
            dl.login_by_token(api_token=token)
        except Exception as e:
            print(f"⚠️ FinMind token 登入失敗，改用匿名額度：{e}")
    return dl


def _is_rate_limit_error(e) -> bool:
    """判斷例外是否為 FinMind 額度用盡（HTTP 402 / 429 或訊息中含相關關鍵字）"""
    msg = str(e)
    return ('402' in msg or '429' in msg or 'Payment Required' in msg
            or 'rate limit' in msg.lower() or '額度' in msg)


class StockDataLoader:
    """台股數據引擎 - 全面採用 FinMind（不依賴 yfinance，避免雲端環境當機風險）"""
    
    def __init__(self):
        self.dl = _create_finmind_loader()

    @st.cache_data(ttl=3600)
    def get_combined_data(_self, stock_id, days, use_adjusted=True):
        """完整數據載入流程
        
        Args:
            stock_id: 股票代碼
            days: 載入天數
            use_adjusted: True=還原K線(復權,預設), False=一般K線
        """
        try:
            warnings = []
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=days + 150)
            start_str = start_date.strftime('%Y-%m-%d')

            # ========== 0. 查詢股票名稱 ==========
            stock_name = stock_id
            try:
                stock_info_all = _self.dl.taiwan_stock_info()
                if not stock_info_all.empty:
                    match_info = stock_info_all[stock_info_all['stock_id'] == stock_id]
                    if not match_info.empty:
                        stock_name = match_info.iloc[0].get('stock_name', stock_id)
            except Exception:
                pass

            if stock_name == stock_id:
                stock_name = get_stock_name(stock_id)

            # ========== 1. 股價數據 ==========
            # 重要：全面改用 FinMind（一般股價 taiwan_stock_daily／還原股價 taiwan_stock_daily_adj），
            # 不再呼叫 yfinance。過去 yfinance 底層的 curl_cffi 元件在雲端環境上曾多次造成整個
            # App 直接當機（Segmentation fault），且 try/except 完全無法攔截這種錯誤，
            # 改用 FinMind 可以徹底排除這個風險來源。

            df = None

            if use_adjusted:
                try:
                    df_adj = _self.dl.taiwan_stock_daily_adj(stock_id=stock_id, start_date=start_str)
                    if df_adj is not None and not df_adj.empty:
                        df = df_adj.rename(columns={
                            'Trading_Volume': 'volume',
                            'max': 'high',
                            'min': 'low'
                        })[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        df['volume'] = (df['volume'] / 1000).astype(int)
                        print("✅ 還原K線：FinMind taiwan_stock_daily_adj")
                except Exception as e:
                    print(f"⚠️ FinMind 還原股價載入失敗，改用一般股價：{e}")
                    df = None

            if df is None:
                df_price = _self.dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_str)

                if df_price.empty:
                    return None, "❌ 查無資料（此股票可能為興櫃、已下市或代碼有誤）", None, []

                df = df_price.rename(columns={
                    'Trading_Volume': 'volume',
                    'max': 'high',
                    'min': 'low'
                })[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

                df['date'] = pd.to_datetime(df['date']).dt.date
                df['volume'] = (df['volume'] / 1000).astype(int)
                if use_adjusted:
                    print("⚠️ 還原股價無資料，改用一般股價（未還原）")
                else:
                    print("✅ 一般K線：FinMind taiwan_stock_daily")

            # ========== 3. 均線 ==========
            for period in [5, 10, 20, 60, 100, 120, 240]:
                df[f'MA{period}'] = df['close'].rolling(window=period).mean()
            
            # ========== 4. 三大法人 ==========
            try:
                df_inst = _self.dl.taiwan_stock_institutional_investors(
                    stock_id=stock_id, 
                    start_date=start_str
                )
                
                if not df_inst.empty:
                    # 計算淨買賣（股）
                    df_inst['net_buy'] = (df_inst['buy'] - df_inst['sell'])
                    df_inst['date'] = pd.to_datetime(df_inst['date']).dt.date
                    
                    # 透視表
                    df_pivot = df_inst.pivot_table(
                        index='date',
                        columns='name',
                        values='net_buy',
                        aggfunc='sum'
                    ).reset_index()
                    
                    # 股 → 張
                    for col in df_pivot.columns:
                        if col != 'date':
                            df_pivot[col] = (df_pivot[col] / 1000)
                    
                    # 標準化欄位
                    rename_dict = {}
                    for col in df_pivot.columns:
                        col_lower = col.lower()
                        if 'foreign' in col_lower:
                            rename_dict[col] = '外資'
                        elif 'investment' in col_lower or 'trust' in col_lower:
                            rename_dict[col] = '投信'
                        elif 'dealer' in col_lower and 'self' in col_lower:
                            rename_dict[col] = '自營商'
                    
                    df_pivot.rename(columns=rename_dict, inplace=True)

                    # ✅ 防呆：rename 後可能產生「重複欄名」，會讓後續 pd.to_numeric 爆掉
                    if df_pivot.columns.duplicated().any():
                        date_part = df_pivot[['date']]
                        num_part = df_pivot.drop(columns=['date'])
                        # 重複欄名合併加總（同一法人不同名稱歸類到同一欄時）
                        num_part = num_part.groupby(level=0, axis=1).sum()
                        df_pivot = pd.concat([date_part, num_part], axis=1)

                    
                    # 主力合計
                    main_cols = [c for c in ['外資', '投信', '自營商'] if c in df_pivot.columns]
                    if main_cols:
                        df_pivot['主力合計'] = df_pivot[main_cols].sum(axis=1)
                    
                    df = pd.merge(df, df_pivot, on='date', how='left')
                else:
                    warnings.append('institutional_empty')
                    
            except Exception as e:
                print(f"法人數據錯誤: {e}")
                if _is_rate_limit_error(e):
                    warnings.append('institutional_rate_limit')
                else:
                    warnings.append('institutional_error')
            
            # ========== 5. 融資融券 ==========
            try:
                df_margin = _self.dl.taiwan_stock_margin_purchase_short_sale(
                    stock_id=stock_id,
                    start_date=start_str
                )
                
                if not df_margin.empty:
                    df_margin['date'] = pd.to_datetime(df_margin['date']).dt.date
                    
                    margin_data = df_margin[['date', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']].copy()
                    margin_data.rename(columns={
                        'MarginPurchaseTodayBalance': '融資餘額',
                        'ShortSaleTodayBalance': '融券餘額'
                    }, inplace=True)
                    
                    margin_data['融資餘額'] = pd.to_numeric(margin_data['融資餘額'], errors='coerce')
                    margin_data['融券餘額'] = pd.to_numeric(margin_data['融券餘額'], errors='coerce')
                    
                    df = pd.merge(df, margin_data, on='date', how='left')
                else:
                    warnings.append('margin_empty')
                    
            except Exception as e:
                print(f"融資數據錯誤: {e}")
                if _is_rate_limit_error(e):
                    warnings.append('margin_rate_limit')
                else:
                    warnings.append('margin_error')
            
            # ========== 6. 數據清洗 ==========
            # 填補0
            fill_cols = ['volume', '外資', '投信', '自營商', '主力合計']
            for col in fill_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # ✅ 防呆：若合併後仍有重複欄名，先處理掉（避免 pd.to_numeric 收到 DataFrame）
            if df.columns.duplicated().any():
                # 同名欄位以加總合併（常見於三大法人欄位 rename 撞名）
                df = df.groupby(level=0, axis=1).sum()

            # 強制轉數值
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          '外資', '投信', '自營商', '主力合計', '融資餘額', '融券餘額']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # ========== 7. 最終輸出 ==========
            df = df.sort_values('date').tail(days).reset_index(drop=True)
            
            # 除錯
            k_type = "還原K線(復權)" if use_adjusted else "一般K線(未復權)"
            print(f"\n【數據載入成功】{stock_id} {stock_name} - {k_type}")
            print(f"資料筆數: {len(df)}")
            if '外資' in df.columns:
                print(f"外資欄位類型: {df['外資'].dtype}")
                print(f"最後3筆外資數據: {df['外資'].tail(3).tolist()}")
            
            return df, None, stock_name, warnings
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"系統錯誤: {str(e)}", None, []
    
    @st.cache_data(ttl=3600)
    def get_monthly_revenue(_self, stock_id):
        """載入近2-3年月營收數據"""
        try:
            # 計算起始日期（往前推36個月）
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=1095)  # 約3年
            start_str = start_date.strftime('%Y-%m-%d')
            
            df_revenue = _self.dl.taiwan_stock_month_revenue(
                stock_id=stock_id,
                start_date=start_str
            )
            
            if df_revenue.empty:
                return None, "查無月營收資料"
            
            # 整理欄位
            df_revenue = df_revenue[['revenue_year', 'revenue_month', 'revenue']].copy()
            df_revenue.columns = ['年', '月', '營收']
            
            # 建立日期欄位（用於繪圖）
            df_revenue['日期'] = pd.to_datetime(
                df_revenue['年'].astype(str) + '-' + df_revenue['月'].astype(str).str.zfill(2) + '-01'
            )
            
            # 轉換營收單位為千元（原始為千元）
            df_revenue['營收'] = pd.to_numeric(df_revenue['營收'], errors='coerce')
            
            # 計算年增率（YoY）
            df_revenue = df_revenue.sort_values('日期').reset_index(drop=True)
            df_revenue['年增率'] = df_revenue['營收'].pct_change(periods=12) * 100
            
            # 只保留最近36個月
            df_revenue = df_revenue.tail(36)
            
            return df_revenue, None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"月營收載入錯誤: {str(e)}"
    
    @st.cache_data(ttl=3600)
    def get_holding_shares_ratio(_self, stock_id):
        """
        載入近2年「股權分散表」，計算大戶（持股 >400張，即 >400,000股）持股比例週趨勢。
        資料來源：集保結算所，FinMind 每週更新一次。
        回傳 DataFrame 欄位：[date, 大戶持股比例]（%），若查無資料回傳 (None, 錯誤訊息)。
        """
        import re
        try:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=730)  # 約2年
            start_str = start_date.strftime('%Y-%m-%d')

            df_holding = _self.dl.taiwan_stock_holding_shares_per(
                stock_id=stock_id,
                start_date=start_str
            )

            if df_holding is None or df_holding.empty:
                return None, "查無股權分散表資料"

            level_col = 'HoldingSharesLevel' if 'HoldingSharesLevel' in df_holding.columns else None
            if level_col is None or 'percent' not in df_holding.columns or 'date' not in df_holding.columns:
                return None, "股權分散表欄位格式異常"

            def _lower_bound(level_str):
                """從級距字串（如 '400,001-600,000' 或 '1,000,001以上'）擷取下界股數"""
                if not isinstance(level_str, str):
                    return None
                nums = re.findall(r'[\d,]+', level_str)
                if not nums:
                    return None
                try:
                    return int(nums[0].replace(',', ''))
                except ValueError:
                    return None

            df_holding = df_holding.copy()
            df_holding['_lower'] = df_holding[level_col].apply(_lower_bound)

            # 大戶定義：持股超過 400 張（400,000 股）以上的級距
            df_major = df_holding[df_holding['_lower'] >= 400001].copy()
            if df_major.empty:
                return None, "查無大戶（400張以上）級距資料"

            df_major['percent'] = pd.to_numeric(df_major['percent'], errors='coerce')
            df_ratio = df_major.groupby('date', as_index=False)['percent'].sum()
            df_ratio.columns = ['date', '大戶持股比例']
            df_ratio['date'] = pd.to_datetime(df_ratio['date'])
            df_ratio = df_ratio.sort_values('date').reset_index(drop=True)

            return df_ratio, None

        except Exception as e:
            import traceback
            traceback.print_exc()
            if _is_rate_limit_error(e):
                return None, "FinMind API 額度已用盡，請稍後再試（免費額度為 300次/小時）"
            return None, f"股權分散表載入錯誤: {str(e)}"

    @st.cache_data(ttl=3600)
    def get_quarterly_data(_self, stock_id):
        """載入近3年季度財務數據（季營收、季毛利率）

        為了避免不同資料源的「type」欄位格式不一致（例如：Q1/Q2、季報、Quarter 等），
        這裡採用「先寬鬆取回 → 再用規則辨識季度」的方式，提高成功率。
        """
        try:
            import re
            # 取回近 3 年資料（約 12 季 + buffer）
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=1200)
            start_str = start_date.strftime('%Y-%m-%d')

            df_fin = _self.dl.taiwan_stock_financial_statement(
                stock_id=stock_id,
                start_date=start_str
            )

            if df_fin is None or df_fin.empty:
                return None, "查無季度財報資料"

            # ===== 0) 判斷是否金融股（避免把一般公司邏輯套到金融股）=====
            def _is_financial_stock(_sid: str) -> bool:
                try:
                    info = _self.dl.taiwan_stock_info()
                    if info is not None and not info.empty and 'stock_id' in info.columns:
                        m2 = info[info['stock_id'] == _sid]
                        if not m2.empty:
                            row = m2.iloc[0].to_dict()
                            # 嘗試從可能的產業欄位判斷
                            for k in ['industry_category', 'industry', 'category', 'type', '產業別', '產業類別', '產業分類', 'industry_category_zh']:
                                if k in row and row[k] is not None:
                                    s = str(row[k])
                                    if any(w in s for w in ['金融', '保險', '金控', '銀行', '證券']):
                                        return True
                except Exception:
                    pass
                # 保底：台股金融族群常見代碼前綴
                return str(_sid).startswith(('28', '58'))

            is_finance = _is_financial_stock(stock_id)

            # ===== 金融股：季營收改用「月營收加總」；毛利率不計算 =====
            if is_finance:
                try:
                    df_m, err_m = _self.get_monthly_revenue(stock_id)
                    if err_m is None and df_m is not None and not df_m.empty:
                        df_m = df_m.copy()
                        col_date = '日期' if '日期' in df_m.columns else ('date' if 'date' in df_m.columns else None)
                        col_rev  = '營收' if '營收' in df_m.columns else ('revenue' if 'revenue' in df_m.columns else None)
                        if col_date is not None and col_rev is not None:
                            df_m[col_date] = pd.to_datetime(df_m[col_date], errors='coerce')
                            df_m = df_m.dropna(subset=[col_date]).sort_values(col_date)
                            df_m['_y'] = df_m[col_date].dt.year.astype('int64')
                            df_m['_q'] = (((df_m[col_date].dt.month - 1) // 3) + 1).astype('int64')
                            df_m[col_rev] = pd.to_numeric(df_m[col_rev], errors='coerce')
                            qsum = df_m.groupby(['_y', '_q'])[col_rev].sum().reset_index()
                            qsum = qsum.rename(columns={'_y': '年度', '_q': '季度', col_rev: '營收'})
                            qsum['季度標籤'] = qsum['年度'].astype(str) + 'Q' + qsum['季度'].astype(str)
                            qsum['毛利率'] = pd.NA
                            qsum['毛利率名稱'] = '毛利率'
                            qsum['是否金融股'] = True
                            return qsum, None
                except Exception:
                    # 若月營收加總也失敗，才繼續走下面的原本邏輯（避免整段中斷）
                    pass


            # ===== 除錯資訊（保留，用來判斷 API 欄位格式）=====
            print(f"\n=== 季度財報除錯資訊 ({stock_id}) ===")
            print(f"欄位: {df_fin.columns.tolist()}")
            print(f"總筆數: {len(df_fin)}")

            # ===== 1) 先嘗試辨識「季度」資料 =====
            df_work = df_fin.copy()

            # 有些資料會用 type 表示季度/年度；先把 type 轉成字串便於判斷
            if 'type' in df_work.columns:
                df_work['type'] = df_work['type'].astype(str)
                type_uniques = sorted(df_work['type'].dropna().unique().tolist())
                print(f"type 唯一值(前 20): {type_uniques[:20]}")

                # 常見季度型態：Q1/Q2/Q3/Q4、1Q/2Q...、季報、Quarter、季
                q_mask = df_work['type'].str.contains(r"(?:^Q[1-4]$|^[1-4]Q$|季|季報|quarter)", case=False, na=False)
                df_q = df_work[q_mask].copy()

                # 若過濾後反而全空，代表 type 不是這種格式（例如根本沒有區分），就退回用全量資料
                if not df_q.empty:
                    df_work = df_q
                    print(f"✓ 以 type 規則辨識季度後筆數: {len(df_work)}")
                else:
                    print("⚠️ type 欄位未能辨識季度格式，改用全量資料繼續嘗試（避免誤殺）")

            # ===== 2) Pivot：date x 科目 =====
            need_cols = {'date', 'origin_name', 'value'}
            if not need_cols.issubset(set(df_work.columns)):
                # 缺欄位就直接回報，並附上目前欄位，方便定位
                return None, f"季度財報欄位不足（需要 date/origin_name/value），目前只有: {', '.join(df_work.columns.astype(str).tolist()[:20])}"

            df_pivot = df_work.pivot_table(
                index=['date'],
                columns='origin_name',
                values='value',
                aggfunc='first'
            ).reset_index()

            # date 轉時間
            df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors='coerce')
            df_pivot = df_pivot[df_pivot['date'].notna()].copy()
            if df_pivot.empty:
                return None, "季度財報日期欄位無法解析"

            # ===== 3) 建立季度標籤 =====
            df_quarterly = pd.DataFrame()
            df_quarterly['年度'] = df_pivot['date'].dt.year
            df_quarterly['季度'] = ((df_pivot['date'].dt.month - 1) // 3) + 1
            df_quarterly['季度標籤'] = df_quarterly['年度'].astype(int).astype(str) + 'Q' + df_quarterly['季度'].astype(int).astype(str)

            # ===== 4) 找「營收」欄位（一般公司優先；金融股/金控用月營收加總作為季度營收）=====
            is_finance = False
            revenue_candidates = []
            for col in df_pivot.columns:
                c = str(col)
                if any(k in c for k in ['營業收入', '收入合計', '營收']) or re.search(r"\brevenue\b", c, re.I):
                    revenue_candidates.append(col)

            # 金融/保險常見的「營收代理」欄位（不一定等於營收，但可用來判斷是否為金融股）
            finance_candidates = []
            for col in df_pivot.columns:
                c = str(col)
                if any(k in c for k in ['淨收益', '利息淨收益', '利息以外淨收益', '保險負債準備淨變動']) or re.search(r"interest\s*net\s*income|net\s*interest|net\s*revenue", c, re.I):
                    finance_candidates.append(col)

            if revenue_candidates:
                rev_col = revenue_candidates[0]
                print(f"✓ 營收欄位(一般): {rev_col}")
                df_quarterly['營收'] = pd.to_numeric(df_pivot[rev_col], errors='coerce')
            else:
                # 找不到一般營收欄位：很可能是金融股/金控
                is_finance = True if finance_candidates else True
                # 先用財報中的代理欄位墊底（避免空值），後續會用「月營收加總」覆蓋季度營收
                if finance_candidates:
                    rev_col = finance_candidates[0]
                    print(f"✓ 營收欄位(金融代理): {rev_col}")
                    df_quarterly['營收'] = pd.to_numeric(df_pivot[rev_col], errors='coerce')
                else:
                    df_quarterly['營收'] = pd.NA
                    print("⚠️ 財報找不到一般營收欄位，改用月營收加總計算季度營收")

            # 金融股：季度營收一律以「月營收 3 個月加總」為準（對齊看盤軟體的季營收）
            if is_finance:
                df_month, _merr = _self.get_monthly_revenue(stock_id)
                if df_month is not None and not df_month.empty:
                    dfm = df_month[['年', '月', '營收']].copy()
                    dfm['日期'] = pd.to_datetime(dfm['年'].astype(str) + '-' + dfm['月'].astype(int).astype(str).str.zfill(2) + '-01', errors='coerce')
                    dfm = dfm[dfm['日期'].notna()].copy()
                    dfm['年度'] = dfm['日期'].dt.year.astype(int)
                    dfm['季度'] = (((dfm['日期'].dt.month - 1) // 3) + 1).astype(int)
                    qsum = dfm.groupby(['年度', '季度'], as_index=False)['營收'].sum()
                    # 用字串鍵合併，避免 pandas 在不同平台發生 int/int64 factorize mismatch
                    df_quarterly['yq_key'] = df_quarterly['年度'].astype(int).astype(str) + 'Q' + df_quarterly['季度'].astype(int).astype(str)
                    qsum['yq_key'] = qsum['年度'].astype(int).astype(str) + 'Q' + qsum['季度'].astype(int).astype(str)
                    df_quarterly = df_quarterly.merge(qsum[['yq_key', '營收']].rename(columns={'營收': '營收_月加總'}), on='yq_key', how='left')
                    df_quarterly['營收'] = pd.to_numeric(df_quarterly['營收_月加總'], errors='coerce').fillna(pd.to_numeric(df_quarterly['營收'], errors='coerce'))
                    df_quarterly = df_quarterly.drop(columns=['營收_月加總'])
                else:
                    print(f"⚠️ 月營收加總失敗: {_merr}")

            # 預設指標名稱
            df_quarterly['毛利率名稱'] = '毛利率'
            # ===== 5) 毛利率：優先用毛利，沒有就用(營收-成本) =====
            # 金融股：不計算毛利率，改用稅後純益率(%) 取代；若算不出則留空
            if is_finance:
                net_col = None
                for col in df_pivot.columns:
                    c = str(col)
                    if any(k in c for k in ['本期稅後淨利', '稅後淨利', '淨利（淨損）', '繼續營業單位本期淨利']) or re.search(r"income\s*after\s*tax|net\s*income", c, re.I):
                        net_col = col
                        break
                if net_col is not None:
                    net_income = pd.to_numeric(df_pivot[net_col], errors='coerce')
                    df_quarterly['毛利率'] = (net_income / pd.to_numeric(df_quarterly['營收'], errors='coerce') * 100).round(2)
                    df_quarterly['毛利率名稱'] = '稅後純益率'
                    print(f"✓ 金融股：以稅後純益率取代毛利率（欄位: {net_col}）")
                else:
                    df_quarterly['毛利率'] = float('nan')
                    df_quarterly['毛利率名稱'] = '稅後純益率'
                    print("⚠️ 金融股：找不到稅後淨利欄位，稅後純益率留空")
            
            # 一般公司：照舊計算毛利率
            gp_col = None
            for col in df_pivot.columns:
                c = str(col)
                if any(k in c for k in ['毛利', '營業毛利']) or re.search(r"gross\s*profit", c, re.I):
                    gp_col = col
                    break

            if gp_col is not None:
                print(f"✓ 毛利欄位: {gp_col}")
                gp = pd.to_numeric(df_pivot[gp_col], errors='coerce')
                df_quarterly['毛利率'] = (gp / df_quarterly['營收'] * 100).round(2)
            else:
                cost_col = None
                for col in df_pivot.columns:
                    c = str(col)
                    if any(k in c for k in ['營業成本', '成本合計']) or re.search(r"cost\s+of\s+revenue|cost\s+of\s+goods", c, re.I):
                        cost_col = col
                        break

                if cost_col is not None:
                    print(f"✓ 成本欄位: {cost_col}")
                    cost = pd.to_numeric(df_pivot[cost_col], errors='coerce')
                    df_quarterly['毛利率'] = ((df_quarterly['營收'] - cost) / df_quarterly['營收'] * 100).round(2)
                else:
                    df_quarterly['毛利率'] = float('nan')
                    print("⚠️ 無法找到毛利/成本欄位，毛利率將顯示空值")

            # ===== 6) 清洗與排序 =====
            df_quarterly = df_quarterly.dropna(subset=['營收']).copy()
            # ✅ 金融股：允許負數營收（投資損失等）；一般公司：過濾負數
            if not is_finance:
                df_quarterly = df_quarterly[df_quarterly['營收'] > 0].copy()
            df_quarterly = df_quarterly.drop_duplicates(subset=['季度標籤'], keep='last')
            df_quarterly = df_quarterly.sort_values(['年度', '季度']).tail(12).reset_index(drop=True)

            if df_quarterly.empty:
                return None, "查無有效季度資料（可能該公司/資料源未提供近年季報）"

            print(f"✓ 成功載入 {len(df_quarterly)} 筆季度資料")
            df_quarterly['是否金融股'] = is_finance
            
            # ✅ 除錯：檢查是否有負數營收
            if (df_quarterly['營收'] < 0).any():
                print(f"⚠️ 發現負數營收（金融股={is_finance}）:")
                neg_data = df_quarterly[df_quarterly['營收'] < 0][['季度標籤', '營收']]
                print(neg_data.to_string(index=False))
            
            return df_quarterly, None

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"載入錯誤: {str(e)}"""