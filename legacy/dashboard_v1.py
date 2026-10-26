#!/usr/bin/env python3
"""
Smart City Energy Command Center - the single live dashboard (Plotly Dash).

    ingestion -> batch layer -> stream + prediction -> NoSQL store -> this dashboard

Batch panels and model metrics are read from the NoSQL store (MongoDB - local server
or Atlas - when MONGODB_URI is set, local JSONL otherwise).  Live panels follow the stream engine;
the alert feed and storage counters are queried from the store itself.

Run:  python dashboard.py     then open http://127.0.0.1:8050
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from urllib.parse import quote

os.environ.setdefault("OMP_NUM_THREADS", "1")   # tiny 40-row predictions: extra threads only add contention
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots

import pipeline
from store import Store
from stream import StreamEngine

# ------------------------------------------------------------------ palette (dark)
BLUE, ORANGE = "#3987e5", "#d95926"
INK1, INK2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS, SURFACE = "#2c2c2a", "#383835", "#1a1a19"
WARN, CRIT, GOOD = "#fab219", "#d03b3b", "#0ca30c"
SEQ_BLUE = [[0, "#0d366b"], [0.25, "#184f95"], [0.5, "#2a78d6"], [0.75, "#6da7ec"], [1, "#cde2fb"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --------------------------------------------------------------------- boot
store = Store()


def ensure_published() -> None:
    """First run against an empty cluster: publish the batch artifacts written by pipeline.py."""
    if store.count("batch_summary") and store.count("model_metrics"):
        return
    agg = json.loads((pipeline.ARTIFACTS / "batch_aggregates.json").read_text(encoding="utf-8"))
    metrics = json.loads((pipeline.ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    store.replace_all("batch_summary", [{"_id": k, "data": v} for k, v in agg.items()])
    store.replace_all("model_metrics", [{"_id": "latest", **metrics}])


ensure_published()
engine = StreamEngine(store)
BATCH = {d["_id"]: d["data"] for d in store.find("batch_summary")}
METRICS = store.get("model_metrics", "latest")
BEST_REG = next(r for r in METRICS["regression"] if r["model"] == METRICS["best_regressor"])
BEST_CLS = next(r for r in METRICS["classification"] if r["model"] == METRICS["best_classifier"])
PERSIST = next(r for r in METRICS["regression"] if r["model"].startswith("Persistence"))
CITIES = sorted(BATCH["cities"], key=lambda c: -c["population"])
DEFAULT_CITY = CITIES[0]["city"]
_g = BATCH["grid_groups"]
GRID_NOTE = (f"Batch layer · high-adoption cities use {abs(_g[-1]['mwh_per_100k'] / _g[0]['mwh_per_100k'] - 1):.0%} "
             f"{'less' if _g[-1]['mwh_per_100k'] < _g[0]['mwh_per_100k'] else 'more'} energy per resident (MWh / 100k)")
_spark = store.get("model_metrics", "spark")
if _spark:
    _sb = min(_spark["regression"], key=lambda r: r["mae"])
    SPARK_NOTE = (f"Spark MLlib {_spark['spark_version']} cross-check: {_sb['model']} MAE {_sb['mae']:.2f} MWh "
                  f"(R² {_sb['r2']:.3f}) vs scikit-learn {BEST_REG['mae']:.2f} — see spark_mllib_train.py")
else:
    SPARK_NOTE = "Spark MLlib comparison in spark_mllib_train.py"


# ------------------------------------------------------------------ helpers
def fmt_ts(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%a %d %b · %H:%M")


def fmt_int(n) -> str:
    return f"{int(round(n)):,}"


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK2, size=12),
        hoverlabel=dict(bgcolor="#262624", bordercolor=AXIS, font=dict(family=FONT, color=INK1, size=12)),
        showlegend=False, **kw)
    return fig


def style_axes(fig: go.Figure, **kw) -> None:
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=11), **kw)
    fig.update_yaxes(gridcolor=GRID, linecolor="rgba(0,0,0,0)", zeroline=False, tickfont=dict(color=MUTED, size=11))


def empty_fig(height: int, title: str, body: str = "") -> go.Figure:
    """Placeholder shown before the stream starts: says what will appear and what to do."""
    fig = go.Figure()
    fig.add_annotation(text=f"<b>{title}</b>", x=0.5, y=0.62 if body else 0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=18, color=INK1))
    if body:
        fig.add_annotation(text=body, x=0.5, y=0.44, xref="paper", yref="paper", showarrow=False,
                           align="center", font=dict(size=13, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base_layout(fig, height)


def spark(values, color=BLUE) -> html.Img:
    if len(values) < 2:
        return html.Div()
    v = np.asarray(values, float)
    lo, hi = v.min(), v.max()
    y = 30 - 26 * ((v - lo) / (hi - lo if hi > lo else 1))
    x = np.linspace(3, 129, len(v))
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 34"><polyline points="{pts}" fill="none" '
           f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
           f'<circle cx="{x[-1]:.1f}" cy="{y[-1]:.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/></svg>')
    return html.Img(src="data:image/svg+xml;utf8," + quote(svg), className="spark")


def card(title, subtitle, child, cls="", right=None) -> html.Div:
    # `is not None`: a Dash component with no children is falsy, so `right or ...` would drop it
    head = html.Div([html.Div([html.H3(title), html.P(subtitle)]), right if right is not None else html.Div()],
                    className="card-head")
    return html.Div([head, child], className=f"card {cls}")


def legend(items) -> html.Div:
    return html.Div([html.Span([html.I(style={"background": c}, className=k), t]) for t, c, k in items],
                    className="legend")


# --------------------------------------------------------- static (batch) figures
def fig_heatmap() -> go.Figure:
    z = np.zeros((7, 24))
    for r in BATCH["heatmap"]:
        z[r["dayofweek"], r["hour"]] = r["mean_mwh"]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}" for h in range(24)], y=DAYS, colorscale=SEQ_BLUE, xgap=2, ygap=2,
        colorbar=dict(thickness=8, len=0.9, tickfont=dict(color=MUTED, size=10), outlinewidth=0, ticksuffix=""),
        hovertemplate="%{y} %{x}:00<br><b>%{z:.0f} MWh</b> average per city<extra></extra>"))
    base_layout(fig, 262)
    style_axes(fig, title=dict(text="hour of day", font=dict(color=MUTED, size=11)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, dtick=3)
    return fig


def fig_grid_groups() -> go.Figure:
    g = BATCH["grid_groups"]
    names = [r["grid_group"].replace(" adoption", "") for r in g]
    vals = [r["mwh_per_100k"] for r in g]
    fig = go.Figure(go.Bar(
        x=vals, y=[f"{n} · {r['cities']} cities" for n, r in zip(names, g)], orientation="h",
        marker=dict(color=BLUE, cornerradius=4), text=[f"{v:.2f}" for v in vals], textposition="outside",
        textfont=dict(color=INK1, size=12), cliponaxis=False,
        hovertemplate="%{y}<br><b>%{x:.1f} MWh</b> per 100k residents<extra></extra>"))
    base_layout(fig, 262, bargap=0.62)
    style_axes(fig)
    fig.update_xaxes(visible=False, range=[0, max(vals) * 1.18])
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK2, size=12))
    return fig


def fig_models() -> go.Figure:
    rows = sorted(METRICS["regression"], key=lambda r: -r["mae"])
    colors = [BLUE if r["model"] == METRICS["best_regressor"] else
              ("#4c4c49" if r["kind"] == "baseline" else "#256abf") for r in rows]
    fig = go.Figure(go.Bar(
        x=[r["mae"] for r in rows], y=[r["model"] for r in rows], orientation="h",
        marker=dict(color=colors, cornerradius=4), cliponaxis=False,
        text=[f"{r['mae']:.2f}" + ("  ✓ selected" if r["model"] == METRICS["best_regressor"] else "") for r in rows],
        textposition="outside", textfont=dict(color=INK1, size=12),
        customdata=[[r["rmse"], r["r2"]] for r in rows],
        hovertemplate="%{y}<br>MAE <b>%{x:.2f} MWh</b><br>RMSE %{customdata[0]:.2f} · R² %{customdata[1]:.3f}<extra></extra>"))
    base_layout(fig, 290, bargap=0.5)
    style_axes(fig)
    fig.update_xaxes(visible=False, range=[0, max(r["mae"] for r in rows) * 1.15])
    fig.update_yaxes(tickfont=dict(color=INK2, size=12))
    return fig


NICE = {"lag_1h": "Load 1 h ago", "lag_24h": "Load 24 h ago", "rolling_24h_mean": "24 h average load",
        "temperature_c": "Temperature", "hour": "Hour of day", "dayofweek": "Day of week",
        "is_weekend": "Weekend", "Population": "Population", "Area (sq. km)": "Area",
        "Smart Infrastructure Score": "Infrastructure score", "Public Transport Usage": "Public transport",
        "Smart Grid Adoption": "Smart grid adoption", "Internet Speed (Mbps)": "Internet speed"}


def fig_importance() -> go.Figure:
    top = METRICS["importance"][:7][::-1]
    fig = go.Figure(go.Bar(
        x=[r["importance"] for r in top], y=[NICE.get(r["feature"], r["feature"]) for r in top], orientation="h",
        marker=dict(color=BLUE, cornerradius=4), cliponaxis=False,
        text=[f"+{r['importance']:.1f}" for r in top], textposition="outside", textfont=dict(color=INK1, size=12),
        hovertemplate="%{y}<br>error rises by <b>%{x:.2f} MWh</b> if shuffled<extra></extra>"))
    base_layout(fig, 290, bargap=0.5)
    style_axes(fig)
    fig.update_xaxes(visible=False, range=[0, max(r["importance"] for r in top) * 1.18])
    fig.update_yaxes(tickfont=dict(color=INK2, size=12))
    return fig


# ----------------------------------------------------------- live figures
def fig_live(snap) -> go.Figure:
    s = snap["series"]
    if not s:
        return empty_fig(330, "Ready when you are  ·  press  ▶ Start live demo",
                         "The system will replay 20 days of city demand it has never seen, one hour at a time.<br>"
                         "Each hour: read the data  →  forecast demand  →  check for spikes  →  save to MongoDB.<br>"
                         "Demand, forecasts and alerts will draw here live.")
    x = [p["timestamp"] for p in s]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x, y=[p["actual"] for p in s], name="Actual", mode="lines",
                             line=dict(color=BLUE, width=2, shape="spline", smoothing=0.5),
                             fill="tozeroy", fillcolor="rgba(57,135,229,0.10)",
                             hovertemplate="Actual <b>%{y:,.0f} MWh</b><extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=[p["predicted"] for p in s], name="Forecast", mode="lines",
                             line=dict(color=ORANGE, width=2, shape="spline", smoothing=0.5),
                             hovertemplate="Forecast <b>%{y:,.0f} MWh</b><extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=[x[-1]], y=[s[-1]["actual"]], mode="markers", hoverinfo="skip",
                             marker=dict(color=BLUE, size=10, line=dict(color=SURFACE, width=2))), 1, 1)
    fig.add_trace(go.Bar(x=x, y=[p["alerts"] for p in s], name="Alerts", marker=dict(color=WARN, cornerradius=3),
                         hovertemplate="<b>%{y}</b> spike alerts<extra></extra>"), 2, 1)
    base_layout(fig, 330, hovermode="x unified", bargap=0.35)
    style_axes(fig)
    fig.update_xaxes(tickformat="%a %H:%M", nticks=6, showgrid=False)
    fig.update_yaxes(tickformat="~s", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title=dict(text="alerts / h", font=dict(color=MUTED, size=11)), row=2, col=1, rangemode="tozero")
    return fig


def fig_risk(snap) -> go.Figure:
    rows = sorted(snap["latest"], key=lambda r: -r["prob"])[:10][::-1]
    if not rows:
        return empty_fig(330, "Spike risk appears here",
                         "The 10 cities most likely to hit<br>a demand spike in the current hour")
    colors = [CRIT if r["prob"] >= 0.85 else WARN if r["prob"] >= 0.5 else BLUE for r in rows]
    n_over = sum(1 for r in snap["latest"] if r["prob"] >= 0.5)
    top = rows[-1]["prob"]
    xmax = max(0.2, min(1.0, top * 1.4))            # scale to the data so quiet hours still read clearly
    fig = go.Figure(go.Bar(
        x=[r["prob"] for r in rows], y=[r["city"] for r in rows], orientation="h",
        marker=dict(color=colors, cornerradius=4), cliponaxis=False,
        text=[f"{r['prob']:.0%}" for r in rows], textposition="outside", textfont=dict(color=INK1, size=12),
        customdata=[[r["predicted"], r["actual"]] for r in rows],
        hovertemplate="%{y}<br>spike risk <b>%{x:.0%}</b><br>forecast %{customdata[0]:.0f} MWh<extra></extra>"))
    if xmax >= 0.5:                                 # alert line only when it falls inside the chart
        fig.add_vline(x=0.5, line=dict(color=MUTED, width=1), opacity=0.7)
    status = (f"▲ {n_over} {'city' if n_over == 1 else 'cities'} above the 50% alert level" if n_over
              else "✓ All cities below the 50% alert level")
    fig.add_annotation(text=status, x=0, y=1.03, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                       showarrow=False, font=dict(size=12.5, color=INK1))
    base_layout(fig, 330, bargap=0.45)
    fig.update_layout(margin=dict(l=4, r=4, t=30, b=4))
    style_axes(fig)
    fig.update_xaxes(visible=False, range=[0, xmax + 0.1 * xmax])
    fig.update_yaxes(tickfont=dict(color=INK2, size=12))
    return fig


def fig_city(snap, city) -> go.Figure:
    h = snap["city_hist"].get(city, [])
    if not h:
        return empty_fig(262, "Pick a city, then press Start",
                         "Its actual and forecast demand will draw here,<br>hour by hour as the stream runs")
    x = [p["timestamp"] for p in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[p["actual"] for p in h], name="Actual", mode="lines",
                             line=dict(color=BLUE, width=2, shape="spline", smoothing=0.5), fill="tozeroy",
                             fillcolor="rgba(57,135,229,0.10)", hovertemplate="Actual <b>%{y:.1f} MWh</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[p["predicted"] for p in h], name="Forecast", mode="lines",
                             line=dict(color=ORANGE, width=2, shape="spline", smoothing=0.5),
                             hovertemplate="Forecast <b>%{y:.1f} MWh</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[h[-1]["actual"]], mode="markers", hoverinfo="skip",
                             marker=dict(color=BLUE, size=10, line=dict(color=SURFACE, width=2))))
    base_layout(fig, 262, hovermode="x unified")
    style_axes(fig)
    fig.update_xaxes(tickformat="%a %H:%M", nticks=4, showgrid=False)
    fig.update_yaxes(rangemode="tozero", tickformat="~s")
    return fig


# --------------------------------------------------------------- components
def kpi(label, value, unit, foot, tag=None, hero=False, sp=None) -> html.Div:
    return html.Div([
        html.Div([html.Span(label), html.Span(tag, className="k-tag") if tag else None], className="k-label"),
        html.Div([value, html.Small(unit) if unit else None], className="k-value"),
        html.Div(foot, className="k-foot"),
        sp if sp is not None else html.Div(),
    ], className="card kpi hero" if hero else "card kpi")


def kpi_row(snap) -> list:
    s = snap["series"]
    if s:
        last = s[-1]
        dev = (last["predicted"] - last["actual"]) / last["actual"]
        load = kpi("Citywide load now", fmt_int(last["actual"]), "MWh",
                   html.Span(["forecast ", html.B(fmt_int(last["predicted"])), " · ",
                              html.Span(f"{dev:+.1%}", className="delta-good" if abs(dev) < 0.03 else "delta-bad"),
                              f" · {last['temperature']:.1f} °C avg"]),
                   tag="live" if snap["running"] else "paused", hero=True, sp=spark([p["actual"] for p in s]))
        mae = kpi("Average forecast miss", f"{snap['mae']:.2f}", "MWh",
                  "how far each hourly forecast is off, per city, on average", tag="MAE")
    else:
        load = kpi("Citywide load now", "—", "MWh", "Press Start to see live demand", tag="idle", hero=True)
        mae = kpi("Average forecast miss", "—", "MWh", "how far each hourly forecast is off, per city, on average",
                  tag="MAE")
    if s:
        real = f"{snap['precision']:.0%}" if snap["precision"] is not None else "—"
        caught = f"{snap['recall']:.0%}" if snap["recall"] is not None else "—"
        alert_foot = html.Span([html.B(real), " of alerts were real spikes · ", html.B(caught), " of spikes caught"])
    else:
        alert_foot = "early warnings for hours of unusually high demand"
    alerts = kpi("Spike alerts raised", fmt_int(snap["alerts"]), None, alert_foot,
                 tag=f"{s[-1]['alerts']} this hour" if s else None,
                 sp=spark([p["alerts"] for p in s], WARN) if s else None)
    quality = kpi("Forecast accuracy", f"{BEST_REG['r2'] * 100:.1f}", "%",
                  html.Span([html.B(METRICS["best_regressor"]), " · ",
                             html.Span(f"{PERSIST['mae'] / BEST_REG['mae']:.1f}× more accurate", className="delta-good"),
                             " than guessing last hour's load"]), tag=f"R² {BEST_REG['r2']:.3f}")
    return [load, mae, alerts, quality]


def stepper(snap) -> list:
    running, fin = snap["running"], snap["finished"]
    started = snap["tick"] > 0
    live_state = "active" if running else ("done" if fin else ("paused" if started else ""))
    items = [
        ("Ingest", f"CSV · {fmt_int(BATCH['kpis']['rows'])} rows", "done"),
        ("Batch", "Parquet + aggregates", "done"),
        ("Stream", f"hour {snap['tick']} / {snap['n_ticks']}", live_state),
        ("Predict", f"{fmt_int(snap['predictions'])} forecasts", "active" if running else ("done" if started else "")),
        ("Store", store.status()["label"] if store.backend == "mongodb" else "Local NoSQL files",
         "active" if running else ("done" if started else "")),
        ("Visualize", "live dashboard", "active" if running else "done"),
    ]
    out = []
    for name, sub, st in items:
        ico = {"done": "✓", "active": "●", "paused": "❚❚"}.get(st, "○")
        out.append(html.Div([html.Div([html.Span(ico, className="s-ico"), html.Span(name)], className="s-top"),
                             html.Div(sub, className="s-sub")], className=f"step {st}"))
    return out


def alert_table(docs) -> html.Div:
    if not docs:
        return html.Div("No spike alerts yet — they appear as the stream runs.", className="empty")
    rows = []
    for d in docs:
        sev = d["severity"].lower()
        icon = "▲" if sev == "high" else "●"
        verdict = (html.Span("✓ spike confirmed", className="chip ok") if d["confirmed"]
                   else html.Span("– no spike", className="chip miss"))
        rows.append(html.Tr([
            html.Td(fmt_ts(d["timestamp"])), html.Td(d["city"], className="city"),
            html.Td(f"{d['spike_probability']:.0%}"), html.Td(f"{d['predicted_mwh']:.0f}"),
            html.Td(html.Span(f"{icon} {d['severity']}", className=f"chip {sev}")), html.Td(verdict)]))
    head = html.Tr([html.Th(t) for t in ("Hour", "City", "Risk", "Forecast MWh", "Severity", "Outcome")])
    return html.Table([html.Thead(head), html.Tbody(rows)], className="alerts")


def storage_panel(status) -> html.Div:
    counts = status["counts"]
    top = max(max(counts.values()), 1)
    rows = [html.Div([html.Span(n.replace("_", " ")),
                      html.Div(html.Div(style={"width": f"{100 * c / top:.0f}%"}), className="bar"),
                      html.Span(f"{c:,}", className="n")], className="store-row") for n, c in counts.items()]
    return html.Div(rows, className="stores")


# ------------------------------------------------------------------- layout
LOGO = html.Img(src="data:image/svg+xml;utf8," + quote(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>'),
    style={"width": "26px", "height": "26px"})

app = Dash(__name__, title="Smart City Energy Command Center", update_title=None)


def serve_layout():
    k = BATCH["kpis"]
    backend_ok = store.backend == "mongodb"
    steps, clock0, prog0, ptext0, kpis0, live0, risk0, city0, start_cls0 = live_outputs(DEFAULT_CITY)
    graph = {"displayModeBar": False}
    return html.Div(className="shell", children=[
        html.Div(className="topbar", children=[
            html.Div(className="brand", children=[
                html.Div(LOGO, className="logo"),
                html.Div([html.H1("Smart City Energy Command Center"),
                          html.P(f"{k['cities']} cities · hourly demand forecasting and spike detection")])]),
            html.Div(className="controls", children=[
                html.Span([html.Span(className="dot on" if backend_ok else "dot warn"),
                           f"{store.status()['label']} connected" if backend_ok else "Local NoSQL store (no database)"],
                          className="badge", title=store.status()["detail"] if backend_ok else store.error),
                dcc.RadioItems(id="speed", options=[{"label": s, "value": s} for s in ("1x", "2x", "5x")],
                               value="1x", className="speed", inline=True,
                               labelStyle={"display": "inline-block"}),
                html.Button("▶  Start live demo", id="btn-start", n_clicks=0, className=start_cls0),
                html.Button("⏸  Pause", id="btn-pause", n_clicks=0, className="btn"),
                html.Button("↺  Reset", id="btn-reset", n_clicks=0, className="btn")])]),

        html.Div(steps, id="stepper", className="stepper"),
        html.Div(className="clockbar", children=[
            html.Div(clock0, id="clock", className="clock"),
            html.Div(html.Div(id="progress-fill", style=prog0), className="progress"),
            html.Div(ptext0, id="ptext", className="ptext")]),

        html.Div(kpis0, id="kpis", className="grid kpis"),

        html.Div(className="grid", children=[
            card("Citywide demand · actual vs forecast", "All 40 cities, one point per streamed hour",
                 dcc.Graph(id="live-chart", figure=live0, config=graph), "c8",
                 right=legend([("Actual", BLUE, ""), ("Forecast", ORANGE, ""), ("Alerts", WARN, "sq")])),
            card("Spike risk · right now", "Top 10 cities · chance of a spike",
                 dcc.Graph(id="risk-chart", figure=risk0, config=graph), "c4",
                 right=legend([("High", CRIT, "sq"), ("Med", WARN, "sq"), ("Low", BLUE, "sq")]))]),

        html.Div(className="grid", children=[
            card("Weekly demand rhythm", "Batch layer · average MWh per city by weekday and hour",
                 dcc.Graph(figure=fig_heatmap(), config={"displayModeBar": False}), "c5"),
            card("City explorer", "Follow any city through the stream",
                 dcc.Graph(id="city-chart", figure=city0, config=graph), "c4",
                 right=dcc.Dropdown(id="city", className="city-select", clearable=False, value=DEFAULT_CITY,
                                    options=[{"label": f"{c['city']} · {c['population'] / 1e6:.1f}M", "value": c["city"]}
                                             for c in CITIES])),
            card("Smart grid effect", GRID_NOTE,
                 dcc.Graph(figure=fig_grid_groups(), config={"displayModeBar": False}), "c3")]),

        html.Div(className="grid", children=[
            card("Forecast models compared", "Average miss (MAE, in MWh) on 20 unseen days · shorter bar = more accurate",
                 dcc.Graph(figure=fig_models(), config={"displayModeBar": False}), "c4"),
            card("What drives demand", "How much worse the forecast gets (MWh) if this input is hidden",
                 dcc.Graph(figure=fig_importance(), config={"displayModeBar": False}), "c4"),
            card("Spike alert quality", "Raises an alert when a city's spike chance reaches 50%",
                 html.Div(className="stores wide", children=[
                     html.Div([html.Span("Alerts that were real"), html.Div(html.Div(style={"width": f"{BEST_CLS['precision'] * 100:.0f}%"}), className="bar"),
                               html.Span(f"{BEST_CLS['precision']:.0%}", className="n")], className="store-row"),
                     html.Div([html.Span("Spikes caught"), html.Div(html.Div(style={"width": f"{BEST_CLS['recall'] * 100:.0f}%"}), className="bar"),
                               html.Span(f"{BEST_CLS['recall']:.0%}", className="n")], className="store-row"),
                     html.Div([html.Span("Overall balance (F1)"), html.Div(html.Div(style={"width": f"{BEST_CLS['f1'] * 100:.0f}%"}), className="bar"),
                               html.Span(f"{BEST_CLS['f1']:.2f}", className="n")], className="store-row"),
                     html.Div([html.Span("Spike vs normal (AUC)"), html.Div(html.Div(style={"width": f"{BEST_CLS['roc_auc'] * 100:.0f}%"}), className="bar"),
                               html.Span(f"{BEST_CLS['roc_auc']:.3f}", className="n")], className="store-row"),
                     html.Div([html.Div([html.Span("Training rows"), html.B(f"{METRICS['train_rows']:,}")]),
                               html.Div([html.Span("Test rows"), html.B(f"{METRICS['test_rows']:,}")]),
                               html.Div([html.Span("Input features"), html.B(str(len(METRICS['importance'])))]),
                               html.Div([html.Span("Alert level"), html.B(f"{METRICS['alert_probability']:.0%} risk")])],
                              className="kv"),
                     html.P("Spike = demand above a city's own 90th percentile in the training period.",
                            className="note")]),
                 "c4")]),

        html.Div(className="grid", children=[
            card("Live alert feed", "Read from the NoSQL alerts collection",
                 html.Div(id="alerts"), "c8"),
            card("NoSQL storage", "Documents per collection", html.Div(id="storage"), "c4",
                 right=html.Span(store.status()["label"] if backend_ok else "Local files", className="badge"))]),

        html.Div(f"Data: {k['rows']:,} hourly readings · {k['period_start'][:10]} to {k['period_end'][:10]} · "
                 f"{SPARK_NOTE}", className="footer"),

        dcc.Interval(id="fast", interval=1000),
        dcc.Interval(id="slow", interval=3000),
        html.Div(id="sink", style={"display": "none"}),
    ])


# ---------------------------------------------------------------- callbacks
@app.callback(Output("sink", "children"), Input("btn-start", "n_clicks"), Input("btn-pause", "n_clicks"),
              Input("btn-reset", "n_clicks"), Input("speed", "value"), prevent_initial_call=True)
def control(_s, _p, _r, speed):
    t = ctx.triggered_id
    if t == "speed":
        engine.set_speed(speed)
    elif t == "btn-start":
        engine.set_speed(speed)
        engine.start()
    elif t == "btn-pause":
        engine.pause()
    elif t == "btn-reset":
        engine.reset()
    return t


def live_outputs(city):
    """Everything that follows the stream; used for the first paint and every refresh."""
    snap = engine.snapshot()
    s = snap["series"]
    pct = 100 * snap["tick"] / snap["n_ticks"]
    clock = (html.Span(["Stream time  ", html.B(fmt_ts(s[-1]["timestamp"]))]) if s
             else html.Span("Stream idle · press Start"))
    state = "running" if snap["running"] else ("complete" if snap["finished"] else ("paused" if s else "ready"))
    ptext = f"{snap['tick']} / {snap['n_ticks']} h · {state}"
    if snap["error"]:
        ptext = "warning: " + snap["error"][:60]
    start_cls = "btn btn-primary" + (" pulse" if snap["tick"] == 0 and not snap["running"] else "")
    return (stepper(snap), clock, {"width": f"{pct:.1f}%"}, ptext, kpi_row(snap),
            fig_live(snap), fig_risk(snap), fig_city(snap, city), start_cls)


@app.callback(
    Output("stepper", "children"), Output("clock", "children"), Output("progress-fill", "style"),
    Output("ptext", "children"), Output("kpis", "children"), Output("live-chart", "figure"),
    Output("risk-chart", "figure"), Output("city-chart", "figure"), Output("btn-start", "className"),
    Input("fast", "n_intervals"), Input("city", "value"))
def refresh_live(_n, city):
    return live_outputs(city)


@app.callback(Output("alerts", "children"), Output("storage", "children"), Input("slow", "n_intervals"))
def refresh_store(_n):
    docs = store.find("alerts", sort="tick", limit=7)
    return alert_table(docs), storage_panel(store.status())


app.layout = serve_layout          # assigned last: the layout function uses live_outputs() above


if __name__ == "__main__":
    import sys
    if "--warmup" in sys.argv:                     # pre-stream N hours (used for screenshots)
        n = int(sys.argv[sys.argv.index("--warmup") + 1])
        engine.reset()
        for _ in range(n):
            engine._process_tick()
        print(f"Warm-up: {n} hours already streamed")
    PORT = int(os.environ.get("PORT", "8050"))
    print(f"\nSmart City Energy Command Center -> http://127.0.0.1:{PORT}")
    print(f"Storage backend: {store.status()['label']}  ({store.status()['detail'] or 'ok'})\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)
