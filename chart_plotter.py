"""圖表層：Plotly，紅漲綠跌，白底統一風格。"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN, ACCENT, GRID = "#d62839", "#2a9d3e", "#5b4b9e", "#eee7f9"
MA_COLORS = {"MA5": "#f4a300", "MA20": "#c2255c", "MA60": "#7c3aed",
             "MA100": "#0891b2", "MA120": "#ea580c", "MA240": "#b91c1c"}
CONFIG = {"displaylogo": False, "scrollZoom": True,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"]}


def _pn(values):
    """正值紅、負值綠。"""
    return [UP if v >= 0 else DOWN for v in values]


def _style(fig, height, title=None):
    fig.update_layout(
        height=height, template="plotly_white", hovermode="x unified",
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        title=dict(text=title, x=0.5, font=dict(color=ACCENT, size=15)) if title else None,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def _skip_non_trading(fig, dates):
    """隱藏非交易日，讓 K 線連續。"""
    d = pd.to_datetime(dates)
    gaps = pd.date_range(d.min(), d.max()).difference(d)
    fig.update_xaxes(rangebreaks=[dict(values=gaps.strftime("%Y-%m-%d").tolist())])


def price_chart(df, title, mas):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=UP, decreasing_line_color=DOWN, name="K線", showlegend=False), 1, 1)
    for ma in mas:
        fig.add_trace(go.Scatter(x=df["date"], y=df[ma], name=ma,
                                 line=dict(color=MA_COLORS[ma], width=1.5)), 1, 1)
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量(張)", showlegend=False,
                         marker_color=_pn(df["close"] - df["open"])), 2, 1)
    _skip_non_trading(fig, df["date"])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(dragmode="pan")
    return _style(fig, 620, title)


def chip_chart(df):
    rows = [c for c in ("外資", "投信", "融資餘額") if c in df and df[c].notna().any()]
    if not rows:
        return None
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{c}（張）" for c in rows])
    for i, c in enumerate(rows, 1):
        if c == "融資餘額":
            fig.add_trace(go.Scatter(x=df["date"], y=df[c], name=c, line=dict(color=ACCENT, width=2)), i, 1)
        else:
            fig.add_trace(go.Bar(x=df["date"], y=df[c], name=c, marker_color=_pn(df[c])), i, 1)
    _skip_non_trading(fig, df["date"])
    fig.update_layout(showlegend=False)
    fig.update_annotations(font=dict(color=ACCENT, size=13))
    return _style(fig, 150 * len(rows) + 60)


def holding_chart(h):
    h = h.tail(52)
    color = UP if h["大戶持股比例"].iloc[-1] >= h["大戶持股比例"].iloc[0] else DOWN
    fig = go.Figure(go.Scatter(x=h["date"], y=h["大戶持股比例"], mode="lines+markers",
                               line=dict(color=color, width=2), marker=dict(size=4),
                               hovertemplate="%{y:.2f}%<extra></extra>"))
    fig.update_yaxes(ticksuffix="%")
    return _style(fig, 300, "大戶持股比例（>400張）週趨勢")


def _bar_line(x, bars, bar_colors, line, bar_name, line_name, title, line_range=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=bars, name=bar_name, marker_color=bar_colors,
                         hovertemplate="%{y:,.2f} 億<extra></extra>"), secondary_y=False)
    if line is not None and pd.Series(line).notna().any():
        fig.add_trace(go.Scatter(x=x, y=line, name=line_name, mode="lines+markers",
                                 line=dict(color=ACCENT, width=2.5),
                                 hovertemplate="%{y:.2f}%<extra></extra>"), secondary_y=True)
        fig.update_yaxes(ticksuffix="%", showgrid=False, range=line_range, secondary_y=True)
    fig.update_yaxes(title_text="億元", secondary_y=False)
    return _style(fig, 380, title)


def revenue_chart(r):
    colors = ["#aaaaaa" if pd.isna(m) else (DOWN if m < 0 else UP) for m in r["月增率"]]
    yoy = r["年增率"]
    pad = max(10, (yoy.max() - yoy.min()) * 0.2) if yoy.notna().any() else 10
    rng = [min(yoy.min() - pad, -10), max(yoy.max() + pad, 10)] if yoy.notna().any() else None
    return _bar_line(r["日期"], r["營收"] / 1e8, colors, yoy, "月營收", "年增率",
                     "月營收（綠=月減）與年增率", rng)


def quarterly_chart(q):
    has_gm = q["毛利率"].notna().any()
    return _bar_line(q["季度"], q["營收"] / 1e8, _pn(q["營收"]), q["毛利率"] if has_gm else None,
                     "季營收", "毛利率", "季營收與毛利率" if has_gm else "季營收（月營收加總）")
