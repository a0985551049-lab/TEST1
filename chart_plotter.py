import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

def _get_gp_range(df_quarterly, pad_abs=8.0, pad_ratio=0.20, pad_cap=18.0, min_span=16.0):
    """毛利率右軸區間：避免波動被誇大，同時允許負毛利率被看見"""
    try:
        s = pd.to_numeric(df_quarterly.get('毛利率'), errors='coerce').dropna()
        if s.empty:
            return None

        vmin, vmax = float(s.min()), float(s.max())
        span = vmax - vmin

        pad = max(pad_abs, span * pad_ratio)
        pad = min(pad, pad_cap)

        rmin = max(-100.0, vmin - pad)
        rmax = min(100.0,  vmax + pad)

        if (rmax - rmin) < min_span:
            mid = (rmax + rmin) / 2.0
            rmin = max(-100.0, mid - min_span/2.0)
            rmax = min(100.0,  mid + min_span/2.0)

        return [rmin, rmax]
    except Exception:
        return None


def _get_revenue_range(revenue_series, pad_ratio=0.15):
    """季營收Y軸範圍：支持負數營收顯示"""
    try:
        # 轉換為千元單位
        values = (revenue_series / 1000).round(0)
        vmin, vmax = float(values.min()), float(values.max())
        
        print(f"\n_get_revenue_range 除錯:")
        print(f"  原始最小值: {vmin:,.0f}")
        print(f"  原始最大值: {vmax:,.0f}")
        
        # 如果有負數，確保下限低於0
        if vmin < 0:
            # 負數區間：給予足夠的顯示空間
            span = vmax - vmin
            pad = span * pad_ratio
            rmin = vmin - pad
            rmax = vmax + pad
            print(f"  有負數！計算範圍: [{rmin:,.0f}, {rmax:,.0f}]")
        else:
            # 全是正數：從0開始
            rmin = 0
            rmax = vmax * (1 + pad_ratio)
            print(f"  全正數，計算範圍: [0, {rmax:,.0f}]")
        
        return [rmin, rmax]
    except Exception as e:
        print(f"  範圍計算失敗: {e}")
        return None


def _get_yoy_range(df_revenue, pad_abs=8.0, pad_ratio=0.25, pad_cap=20.0, min_span=40.0):
    """年增率右軸區間：不要縮太小誇大波動，也不要放太大看起來沒起伏"""
    try:
        col = '年增率' if '年增率' in df_revenue.columns else ('yoy' if 'yoy' in df_revenue.columns else None)
        if col is None:
            return None

        s = pd.to_numeric(df_revenue[col], errors='coerce').dropna()
        if s.empty:
            return None

        vmin, vmax = float(s.min()), float(s.max())
        span = vmax - vmin

        pad = max(pad_abs, span * pad_ratio)
        pad = min(pad, pad_cap)

        rmin, rmax = vmin - pad, vmax + pad

        # 保留 0% 參考線附近空間
        rmin = min(rmin, -5.0)
        rmax = max(rmax,  5.0)

        if (rmax - rmin) < min_span:
            mid = (rmax + rmin) / 2.0
            rmin, rmax = mid - min_span/2.0, mid + min_span/2.0

        return [rmin, rmax]
    except Exception:
        return None



def plot_price_volume_chart(df, stock_id, stock_name, show_ma_dict, k_line_type="一般K線"):
    """
    股價 K 線（含均線）+ 成交量，白底風格，2 子圖。
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    chart_revision = f"{stock_id}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"{stock_id} {stock_name} 股價走勢 ({k_line_type})",
            "成交量 (張)"
        )
    )

    # ========== K線 ==========
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K線',
        increasing_line_color='#d62839',
        decreasing_line_color='#2a9d3e',
        showlegend=False
    ), row=1, col=1)

    # ========== 均線 ==========
    ma_colors = {
        'MA5': '#f4a300',
        'MA20': '#c2255c',
        'MA60': '#7c3aed',
        'MA100': '#0891b2',
        'MA120': '#ea580c',
        'MA240': '#b91c1c',
    }
    for ma_name, show in show_ma_dict.items():
        if show and ma_name in df.columns:
            valid_ma = df[df[ma_name] > 0]
            if not valid_ma.empty:
                fig.add_trace(go.Scatter(
                    x=valid_ma['date'], y=valid_ma[ma_name],
                    name=ma_name,
                    line=dict(color=ma_colors.get(ma_name, '#5b4b9e'), width=1.6),
                    showlegend=True
                ), row=1, col=1)

    # ========== 成交量 ==========
    if 'volume' in df.columns:
        vol_colors = ['#d62839' if c >= o else '#2a9d3e' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df['date'], y=df['volume'], name='成交量',
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

    dt_all = pd.to_datetime(df['date']).dt.date
    missing_days = [d for d in pd.date_range(dt_all.min(), dt_all.max()).date if d not in set(dt_all)]

    total_days = len(df)
    display_days = min(125, total_days)
    initial_range = [df['date'].iloc[-display_days], df['date'].iloc[-1]]

    fig.update_layout(
        height=680,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#333333', size=14),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=60, b=20),
        uirevision=chart_revision,
        modebar=dict(orientation='v', bgcolor='rgba(255,255,255,0.7)', color='#5b4b9e', activecolor='#8a4fae'),
        xaxis=dict(
            rangebreaks=[dict(values=missing_days)],
            range=initial_range if not st.session_state.get(f'__init_range__{stock_id}', False) else None,
            rangeslider=dict(visible=True, thickness=0.04, bgcolor='#f4f1fc', bordercolor='#e3ddf5', borderwidth=1),
            type='date',
            fixedrange=False
        ),
        xaxis2=dict(matches='x', rangebreaks=[dict(values=missing_days)], fixedrange=False),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(color='#333333', size=12)),
        autosize=True,
        dragmode='zoom'
    )
    fig.update_annotations(font=dict(size=13, color='#5b4b9e'))

    st.session_state[f'__init_range__{stock_id}'] = True

    fig.update_yaxes(row=1, col=1, gridcolor='#eee7f9', fixedrange=False)
    fig.update_yaxes(title_text="張", row=2, col=1, gridcolor='#eee7f9', tickformat=',d', fixedrange=False)
    fig.update_xaxes(showgrid=True, gridcolor='#eee7f9')

    return fig


def plot_chip_chart(df, stock_id, stock_name):
    """
    外資 / 投信買賣超，白底風格，2 子圖（獨立於價格圖之外，避免版面過長）。
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    chart_revision = f"{stock_id}_chip"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.5, 0.5],
        subplot_titles=("外資買賣超 (張)", "投信買賣超 (張)")
    )

    if '外資' in df.columns:
        f_colors = ['#d62839' if v >= 0 else '#2a9d3e' for v in df['外資']]
        fig.add_trace(go.Bar(x=df['date'], y=df['外資'], name='外資',
                              marker_color=f_colors, showlegend=False), row=1, col=1)

    if '投信' in df.columns:
        t_colors = ['#d62839' if v >= 0 else '#2a9d3e' for v in df['投信']]
        fig.add_trace(go.Bar(x=df['date'], y=df['投信'], name='投信',
                              marker_color=t_colors, showlegend=False), row=2, col=1)

    dt_all = pd.to_datetime(df['date']).dt.date
    missing_days = [d for d in pd.date_range(dt_all.min(), dt_all.max()).date if d not in set(dt_all)]

    fig.update_layout(
        height=440,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#333333', size=14),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=50, b=20),
        uirevision=chart_revision,
        modebar=dict(orientation='v', bgcolor='rgba(255,255,255,0.7)', color='#5b4b9e', activecolor='#8a4fae'),
        xaxis=dict(rangebreaks=[dict(values=missing_days)], fixedrange=False),
        xaxis2=dict(matches='x', rangebreaks=[dict(values=missing_days)], fixedrange=False),
        autosize=True,
        dragmode='zoom'
    )
    fig.update_annotations(font=dict(size=13, color='#5b4b9e'))

    fig.update_yaxes(title_text="張", row=1, col=1, gridcolor='#eee7f9', tickformat=',d', fixedrange=False)
    fig.update_yaxes(title_text="張", row=2, col=1, gridcolor='#eee7f9', tickformat=',d', fixedrange=False)
    fig.update_xaxes(showgrid=True, gridcolor='#eee7f9')

    return fig


def plot_margin_chart(df, stock_id, stock_name):
    """
    主力法人15日累計 vs 融資餘額，白底風格，單圖雙Y軸。
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    chart_revision = f"{stock_id}_margin"

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    if '主力合計' in df.columns:
        net_15 = df['主力合計'].rolling(15).sum().fillna(0)
        n_colors = ['#d62839' if v >= 0 else '#2a9d3e' for v in net_15]
        fig.add_trace(go.Bar(
            x=df['date'], y=net_15, name='主力15日',
            marker_color=n_colors, showlegend=True
        ), row=1, col=1, secondary_y=False)

    if '融資餘額' in df.columns:
        df_margin = df[df['融資餘額'] > 0].copy()
        if not df_margin.empty:
            fig.add_trace(go.Scatter(
                x=df_margin['date'], y=df_margin['融資餘額'], name='融資',
                mode='lines+markers',
                line=dict(color='#8a4fae', width=2),
                marker=dict(size=4),
                connectgaps=False,
                showlegend=True
            ), row=1, col=1, secondary_y=True)

    dt_all = pd.to_datetime(df['date']).dt.date
    missing_days = [d for d in pd.date_range(dt_all.min(), dt_all.max()).date if d not in set(dt_all)]

    fig.update_layout(
        height=380,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#333333', size=14),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=60, b=20),
        uirevision=chart_revision,
        modebar=dict(orientation='v', bgcolor='rgba(255,255,255,0.7)', color='#5b4b9e', activecolor='#8a4fae'),
        title={'text': f"{stock_id} {stock_name} 主力法人15日累計 vs 融資", 'x': 0.5, 'xanchor': 'center',
               'font': dict(size=15, color='#5b4b9e')},
        xaxis=dict(rangebreaks=[dict(values=missing_days)], fixedrange=False, gridcolor='#eee7f9'),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(color='#333333', size=12)),
        autosize=True,
        dragmode='zoom'
    )

    fig.update_yaxes(title_text="主力15日 (張)", secondary_y=False, gridcolor='#eee7f9', tickformat=',d', fixedrange=False)
    fig.update_yaxes(title_text="融資 (張)", secondary_y=True, gridcolor='#eee7f9', tickformat=',d', fixedrange=False, showgrid=False)

    return fig


def plot_revenue_chart(df_revenue, stock_id, stock_name):
    """
    月營收柱狀圖 + 年增率曲線圖（雙Y軸）
    """
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # ========== 月營收柱狀圖 ==========
    # 顏色規則：月減（MoM < 0）才變藍色；月增/持平為紅色
    df_revenue = df_revenue.copy()
    col_date = '日期' if '日期' in df_revenue.columns else ('date' if 'date' in df_revenue.columns else None)
    col_rev  = '營收' if '營收' in df_revenue.columns else ('revenue' if 'revenue' in df_revenue.columns else None)
    col_mom  = '月增率' if '月增率' in df_revenue.columns else ('mom' if 'mom' in df_revenue.columns else None)

    if col_date is None or col_rev is None:
        raise ValueError("月營收資料缺少日期/營收欄位")

    # 若沒有月增率欄位，就用營收自行計算（百分比）
    if col_mom is None:
        df_revenue['__mom'] = pd.to_numeric(df_revenue[col_rev], errors='coerce').pct_change() * 100.0
        col_mom = '__mom'

    mom_series = pd.to_numeric(df_revenue[col_mom], errors='coerce')

    colors = []
    for mom in mom_series:
        if pd.isna(mom):
            colors.append('#aaaaaa')   # 灰（無資料）
        elif mom < 0:
            colors.append('#2a9d3e')   # 綠（月減）
        else:
            colors.append('#d62839')   # 紅（月增或持平）

    # 轉換為千元單位，取整數
    revenue_display = (pd.to_numeric(df_revenue[col_rev], errors='coerce') / 1000).round(0).astype('Int64')
    
    fig.add_trace(go.Bar(
        x=df_revenue[col_date],
        y=revenue_display,
        name='月營收',
        marker_color=colors,
        hovertemplate='<b>%{x|%Y-%m}</b><br>營收: %{y:,d} 千元<extra></extra>',
        yaxis='y',
        showlegend=True
    ), secondary_y=False)
    
    # ========== 年增率曲線圖 ==========
    # 過濾掉 NaN 值
    col_yoy = '年增率' if '年增率' in df_revenue.columns else ('yoy' if 'yoy' in df_revenue.columns else None)
    df_yoy = df_revenue[df_revenue[col_yoy].notna()].copy() if col_yoy else df_revenue.iloc[0:0].copy()
    
    fig.add_trace(go.Scatter(
        x=df_yoy[col_date],
        y=df_yoy[col_yoy],
        name='年增率',
        mode='lines+markers',
        line=dict(color='#8a4fae', width=2.5),
        marker=dict(size=6, color='#8a4fae'),
        hovertemplate='<b>%{x|%Y-%m}</b><br>年增率: %{y:.2f}%<extra></extra>',
        yaxis='y2',
        showlegend=True
    ), secondary_y=True)
    
    # 零軸參考線
    fig.add_hline(y=0, line_dash="dash", line_color="#999999", opacity=0.5, secondary_y=True)
    
    # ========== 版面配置 ==========
    fig.update_layout(
        height=460,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#333333', size=14),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=90, b=40),
        modebar=dict(orientation='v', bgcolor='rgba(255,255,255,0.7)', color='#5b4b9e', activecolor='#8a4fae'),
        title={
            'text': f"{stock_id} {stock_name} 月營收與年增率",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='#5b4b9e')
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='#333333', size=12)
        )
    )
    
    # Y軸格式
    fig.update_yaxes(
        title_text="營收 (千元)",
        secondary_y=False,
        gridcolor='#eee7f9',
        tickformat=',d',
        fixedrange=False
    )
    
    fig.update_yaxes(
        title_text="年增率 (%)",
        secondary_y=True,
        gridcolor='#eee7f9',
        tickformat='.1f',
        fixedrange=False,
        showgrid=False
    ,
        range=_get_yoy_range(df_revenue)
    )
    
    # X軸格式
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#eee7f9',
        dtick="M3",  # 每3個月顯示一次刻度
        tickformat="%Y-%m"
    )
    
    return fig


def plot_quarterly_chart(df_quarterly, stock_id, stock_name):
    """
    季營收柱狀圖 + 季毛利率曲線圖（雙Y軸）
    """
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # ✅ 除錯：檢查原始營收數據
    print(f"\n=== 季營收圖表除錯 ({stock_id}) ===")
    print(f"原始營收數據（千元）:")
    print(df_quarterly[['季度標籤', '營收']].to_string(index=False))
    print(f"營收最小值: {df_quarterly['營收'].min():,.0f} 千元")
    print(f"營收最大值: {df_quarterly['營收'].max():,.0f} 千元")
    print(f"有負數: {(df_quarterly['營收'] < 0).any()}")
    
    # 轉換單位：除以1000取整數（支持負數）
    revenue_display = (df_quarterly['營收'] / 1000).round(0).astype('Int64')
    
    # ✅ 確保負數正確轉換（Int64 可能有問題，改用 float）
    revenue_values = revenue_display.astype(float).tolist()
    
    print(f"\n轉換後顯示值:")
    print(f"  季度: {df_quarterly['季度標籤'].tolist()}")
    print(f"  數值: {revenue_values}")
    print(f"  顯示值最小: {min(revenue_values):,.0f}")
    print(f"  顯示值最大: {max(revenue_values):,.0f}")
    print(f"  Y值數據類型: {type(revenue_values[0])}")
    
    # ★★★ 檢查哪些是負數
    negative_indices = [i for i, v in enumerate(revenue_values) if v < 0]
    if negative_indices:
        print(f"\n⚠️ 發現負數營收:")
        for idx in negative_indices:
            print(f"    {df_quarterly['季度標籤'].iloc[idx]}: {revenue_values[idx]:,.0f} 千元")
    
    print("=" * 50)
    
    # ✅ 負數營收用綠色標示，正數用紅色（與全站紅漲綠跌一致）
    colors = ['#2a9d3e' if val < 0 else '#d62839' for val in revenue_values]

    # 金融股：若毛利率欄位不存在或全是空值，標題不顯示「與毛利率」，並加小字備註
    has_gm = ('毛利率' in df_quarterly.columns) and (pd.to_numeric(df_quarterly.get('毛利率'), errors='coerce').notna().any())
    title_text = '季營收柱狀圖 + 季毛利率曲線圖（雙Y軸）' if has_gm else '季營收柱狀圖'

    
    # ========== 季營收柱狀圖 ==========
    fig.add_trace(go.Bar(
        x=df_quarterly['季度標籤'],
        y=revenue_values,  # ✅ 使用 float list，確保負數正確傳入
        name='季營收',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>營收: %{y:,.0f} 千元<extra></extra>',
        yaxis='y',
        showlegend=True
    ), secondary_y=False)
    
    # ★★★ 強制設置 base=0（確保負數柱子向下延伸）
    fig.update_traces(base=0, selector=dict(name='季營收'))
    
    # ✅ 驗證 trace 設置
    print(f"\n季營收柱狀圖驗證:")
    for trace in fig.data:
        if trace.name == '季營收':
            print(f"  trace.type: {trace.type}")
            print(f"  trace.base: {trace.base}")
            print(f"  trace.y (前3個): {trace.y[:3] if len(trace.y) > 3 else trace.y}")
            print(f"  trace.y (負數): {[y for y in trace.y if y < 0]}")
            break
    
    # ========== 毛利率曲線圖 ==========
    # 過濾掉 NaN 值
    df_gp = df_quarterly[df_quarterly['毛利率'].notna()].copy()
    
    gp_available = (not df_gp.empty)

    if not df_gp.empty:
        fig.add_trace(go.Scatter(
            x=df_gp['季度標籤'],
            y=df_gp['毛利率'],
            name='毛利率',
            mode='lines+markers',
            line=dict(color='#8a4fae', width=2.5),
            marker=dict(size=7, color='#8a4fae'),
            hovertemplate='<b>%{x}</b><br>毛利率: %{y:.2f}%<extra></extra>',
            yaxis='y2',
            showlegend=True
        ), secondary_y=True)
    
    # ✅ 零軸參考線（當有負數營收時顯示）- 改為總是顯示
    fig.add_hline(y=0, line_dash="solid", line_color="#999999", opacity=0.6, line_width=2, secondary_y=False)
    
    # ========== 版面配置 ==========
    # ✅ 計算Y軸範圍（在layout之前）
    y_range = _get_revenue_range(df_quarterly['營收'])
    
    fig.update_layout(
        height=400,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#333333', size=14),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=90, b=40),
        modebar=dict(orientation='v', bgcolor='rgba(255,255,255,0.7)', color='#5b4b9e', activecolor='#8a4fae'),
        title={
            'text': (f"{stock_id} {stock_name} 季營收與毛利率" if gp_available else f"{stock_id} {stock_name} 季營收"),
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16, color='#5b4b9e')
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='#333333', size=12)
        )
    )
    
    # ✅ Y軸格式：主軸（營收）- 必須最後設定，確保不被覆蓋
    update_dict = {
        'title_text': "營收 (千元)",
        'gridcolor': '#eee7f9',
        'tickformat': ',d',
        'fixedrange': False,
        'zeroline': True,  # ✅ 顯示零軸線
        'zerolinewidth': 2,
        'zerolinecolor': 'rgba(90,70,180,0.4)',
        'showline': True,
        'linewidth': 1,
        'linecolor': '#cccccc'
    }
    
    # ★★★ 關鍵：明確設置 Y軸範圍，不使用 rangemode
    if y_range:
        update_dict['range'] = y_range
        update_dict['autorange'] = False  # 禁止自動範圍
    else:
        # 如果沒有計算出範圍，允許自動調整但確保包含0
        update_dict['rangemode'] = 'tozero'
    
    fig.update_yaxes(secondary_y=False, **update_dict)
    
    # Y軸格式：副軸（毛利率）
    fig.update_yaxes(
        title_text="毛利率 (%)" if gp_available else "",
        secondary_y=True,
        gridcolor='#eee7f9',
        tickformat='.1f',
        fixedrange=False,
        showgrid=False,
        visible=gp_available,
        range=_get_gp_range(df_quarterly) if gp_available else None
    )
    
    # X軸格式
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#eee7f9'
    )
    
    return fig
