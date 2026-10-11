"""Visual decision support -- Plotly charts and Folium maps.

Colour policy (kept in one place so every figure reads as one system):

  * Categorical identity uses a fixed slot order -- blue, orange, aqua -- never
    cycled. Slot 1 is always "our optimised result", slot 2 always "the
    existing/benchmark result", so a colour means the same thing on every chart.
  * Magnitude uses a single-hue light->dark blue ramp (never a rainbow).
  * Covered/uncovered is a reserved status pair and always ships with a label,
    never colour alone.

Every chart is paired with its underlying table in the app, which satisfies the
contrast-relief requirement for the lighter slots.
"""

from __future__ import annotations

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster

# --- categorical slots (fixed order) ---------------------------------------
SERIES = {
    "light": {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"},
    "dark": {1: "#3987e5", 2: "#d95926", 3: "#199e70"},
}
# --- reserved status colours -----------------------------------------------
STATUS = {
    "light": {"good": "#008300", "critical": "#e34948", "warning": "#eda100"},
    "dark": {"good": "#008300", "critical": "#e66767", "warning": "#c98500"},
}
# --- single-hue sequential ramp, lightness monotone ------------------------
# "Monotone" has to mean monotone *in contrast against the surface it is drawn
# on*, not monotone in absolute lightness. Light-to-dark reads correctly on a
# white page, but on the dark basemap and the dark plot ground the same ramp
# paints the emptiest zones nearly white and the busiest ones almost into the
# background -- the hierarchy inverts. The dark mode therefore runs the same
# eight hues the other way, so in both modes the strongest value is the one
# that stands furthest from its backdrop.
SEQ_BLUE = ["#e8f1fb", "#c3dbf4", "#98c0ec", "#6aa2e0", "#3f85d3",
            "#2a78d6", "#1d5da6", "#133f73"]
SEQ = {"light": SEQ_BLUE, "dark": SEQ_BLUE[::-1]}

# Streamlit loads Inter for the app chrome (see .streamlit/config.toml); naming
# it here too stops every chart from falling back to Plotly's own default and
# rendering in a visibly different typeface from the headings around it.
FONT = ('Inter, "Source Sans Pro", -apple-system, BlinkMacSystemFont, '
        'system-ui, sans-serif')

INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8985",
              "grid": "#e4e3df", "surface": "#fcfcfb"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#8f8e86",
             "grid": "#33322f", "surface": "#1a1a19"},
}


def theme() -> str:
    """Which mode the viewer is in, so figures are selected rather than flipped."""
    try:
        t = getattr(getattr(st, "context", None), "theme", None)
        if t is not None and getattr(t, "type", None) in ("light", "dark"):
            return t.type
    except Exception:  # pragma: no cover - Streamlit version differences
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:  # pragma: no cover
        pass
    return "light"


def tokens() -> tuple[dict, dict, dict]:
    m = theme()
    return SERIES[m], STATUS[m], INK[m]


def seq() -> list[str]:
    """The sequential ramp oriented for the mode the viewer is in."""
    return SEQ[theme()]


def _style(fig: go.Figure, ink: dict, height: int = 340, showlegend: bool = False) -> go.Figure:
    """Recessive axes, transparent surface, text in ink tokens not series colour.

    Margins are deliberately generous and ``automargin`` is on for both axes:
    tick labels here are ZIP codes and thousands-separated counts, which a tight
    margin silently clips rather than wrapping.
    """
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        margin=dict(l=64, r=28, t=52, b=56),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=ink["secondary"], size=12),
        title=dict(font=dict(family=FONT, color=ink["primary"], size=14.5,
                             weight=600),
                   x=0, xanchor="left", y=0.97, yanchor="top"),
        hoverlabel=dict(font_size=12, font_family=FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=ink["secondary"])),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=ink["grid"],
                     automargin=True, title_standoff=12,
                     tickfont=dict(color=ink["secondary"]))
    fig.update_yaxes(gridcolor=ink["grid"], zeroline=False, linecolor=ink["grid"],
                     automargin=True, title_standoff=12,
                     tickfont=dict(color=ink["secondary"]))
    return fig


def _headroom(values, pad: float = 1.14) -> list[float]:
    """An axis range that leaves room for ``textposition="outside"`` labels.

    Plotly sizes a value axis to the bars alone and then draws the label past
    the end of the longest one, so without this the top bar's number is cut off
    by the plot edge -- the single most visible defect on every ranked chart.
    """
    top = float(max(values)) if len(values) else 1.0
    return [0, (top if top > 0 else 1.0) * pad]


# Every count axis in the app abbreviates the same way. Left to Plotly the
# convention is picked per figure from that figure's own magnitude, so the two
# charts that sit side by side under "When load spikes" labelled the same
# quantity two different ways -- 5000/4000/3000 on the hour chart, 12k/10k/8k
# on the weekday chart beside it. Stating the format makes the pair comparable.
COUNT_TICKS = dict(tickformat="~s")


def _alpha(hex_colour: str, a: float) -> str:
    """A token colour at partial opacity, so a fill stays inside the palette.

    Plotly fills take rgba and the tokens are hex, and hardcoding the rgba
    triple is how a chart ends up painted in light-mode blue on the dark
    ground -- the one thing the colour policy at the top of this file exists
    to prevent.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def chart_height(fig: go.Figure) -> int:
    """The height a figure was styled to, for sizing a table to match it.

    Paired chart/table rows only read as one row if both cards end on the same
    line, and a table's height is a function of its row count, not of the chart
    beside it. ``app.tbl_rows_for`` turns this back into a row cap.
    """
    return int(fig.layout.height or 340)


def _lock_category_order(fig: go.Figure, order, axis: str = "y") -> go.Figure:
    """Pin a categorical axis to the order the data was sorted into.

    Plotly re-sorts string categories alphabetically by default, which silently
    destroys a ranked bar chart. The order has to be stated explicitly.
    """
    kw = dict(type="category", categoryorder="array",
              categoryarray=[str(v) for v in order])
    return fig.update_yaxes(**kw) if axis == "y" else fig.update_xaxes(**kw)


# ---------------------------------------------------------------------------
# Step 1 -- statistics charts
# ---------------------------------------------------------------------------

def chart_busiest_zones(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Magnitude by identity -> horizontal bars, sorted, directly labelled."""
    s, _, ink = tokens()
    d = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["incidents"], y=d["zip_code"].astype(str), orientation="h",
        marker=dict(color=s[1], line=dict(width=0)),
        text=[f"{v:,}" for v in d["incidents"]],
        textposition="outside", textfont=dict(color=ink["secondary"]),
        hovertemplate="ZIP %{y}<br>%{x:,} incidents<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title=f"Top {top_n} ZIP zones by EMS incident count")
    fig.update_xaxes(showgrid=True, gridcolor=ink["grid"], title="incidents",
                     range=_headroom(d["incidents"]), **COUNT_TICKS)
    fig.update_yaxes(showgrid=False, title=None)
    _lock_category_order(fig, d["zip_code"])
    return _style(fig, ink, height=max(340, 28 * len(d) + 122))


def chart_calls_by_hour(df: pd.DataFrame) -> go.Figure:
    """Ordered bins over the day -> bars; the peak bar is labelled."""
    s, _, ink = tokens()
    peak = int(df["incidents"].idxmax())
    labels = ["" if i != peak else f"peak {df['incidents'].iloc[i]:,}" for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df["hour"], y=df["incidents"], marker=dict(color=s[1], line=dict(width=0)),
        text=labels, textposition="outside", textfont=dict(color=ink["secondary"]),
        hovertemplate="%{x}:00–%{x}:59<br>%{y:,} incidents<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="EMS calls by hour of day")
    fig.update_xaxes(title="hour of day", dtick=2)
    # Headroom for the "peak" label, which otherwise clips on the tallest bar.
    fig.update_yaxes(title="incidents", range=_headroom(df["incidents"], 1.20),
                     **COUNT_TICKS)
    return _style(fig, ink)


def chart_calls_by_weekday(df: pd.DataFrame) -> go.Figure:
    """Weekday load. The spread across the week is a few percent, so the bars
    are near enough the same height to be unreadable against the axis alone --
    the busiest and quietest days are called out in text instead.
    """
    s, _, ink = tokens()
    hi = int(df["incidents"].idxmax())
    lo = int(df["incidents"].idxmin())
    labels = [f"{df['incidents'].iloc[i]:,}" if i in (hi, lo) else ""
              for i in range(len(df))]
    fig = go.Figure(go.Bar(
        x=df["weekday"], y=df["incidents"], marker=dict(color=s[1], line=dict(width=0)),
        text=labels, textposition="outside", textfont=dict(color=ink["secondary"]),
        hovertemplate="%{x}<br>%{y:,} incidents<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="EMS calls by day of week")
    fig.update_yaxes(title="incidents", range=_headroom(df["incidents"], 1.20),
                     **COUNT_TICKS)
    return _style(fig, ink)


def chart_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Magnitude across two categorical axes -> single-hue sequential cells."""
    _, _, ink = tokens()
    ramp = seq()
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=[f"{h:02d}" for h in pivot.columns], y=list(pivot.index),
        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)],
        hovertemplate="%{y} %{x}:00<br>%{z:,} incidents<extra></extra>",
        colorbar=dict(title=dict(text="calls", font=dict(color=ink["secondary"])),
                      tickfont=dict(color=ink["secondary"]), thickness=12),
        xgap=2, ygap=2,
    ))
    fig.update_layout(title="When load spikes — weekday × hour")
    fig.update_xaxes(title="hour of day")
    # Plotly draws row 0 at the bottom; reverse so the week reads Mon-first
    # downwards, the way a calendar does.
    fig.update_yaxes(autorange="reversed")
    return _style(fig, ink, height=320)


def chart_daily_trend(df: pd.DataFrame) -> go.Figure:
    """Daily volume, with a truncated final day marked rather than dropped.

    The feed is harvested mid-day, so the last day is a partial count. Drawn
    plainly it looks like a collapse in demand, so the complete days carry the
    solid line and the partial day is shown dashed and labelled.
    """
    s, _, ink = tokens()
    partial = bool(df["partial"].iloc[-1]) if "partial" in df and len(df) else False
    solid = df.iloc[:-1] if partial else df

    fig = go.Figure(go.Scatter(
        x=solid["date"], y=solid["incidents"], mode="lines",
        line=dict(color=s[1], width=2), name="complete days",
        hovertemplate="%{x|%d %b %Y}<br>%{y:,} incidents<extra></extra>",
    ))
    if partial and len(df) >= 2:
        tail = df.iloc[-2:]
        fig.add_trace(go.Scatter(
            x=tail["date"], y=tail["incidents"], mode="lines+markers",
            line=dict(color=ink["muted"], width=2, dash="dot"),
            marker=dict(color=ink["muted"], size=6), name="partial day",
            hovertemplate="%{x|%d %b %Y}<br>%{y:,} incidents "
                          "(day still in progress)<extra></extra>",
        ))
        fig.add_annotation(
            x=df["date"].iloc[-1], y=df["incidents"].iloc[-1],
            text="final day partial", showarrow=True, arrowhead=0,
            arrowcolor=ink["muted"], ax=-6, ay=-30, xanchor="right",
            font=dict(color=ink["muted"], size=11),
        )
    fig.update_layout(title="Daily EMS incident volume", hovermode="x unified")
    fig.update_yaxes(title="incidents")
    return _style(fig, ink, height=280)


def chart_call_types(df: pd.DataFrame) -> go.Figure:
    s, _, ink = tokens()
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["incidents"], y=d["type"], orientation="h",
        marker=dict(color=s[1], line=dict(width=0)),
        text=[f"{v:,}" for v in d["incidents"]], textposition="outside",
        textfont=dict(color=ink["secondary"]),
        hovertemplate="%{y}<br>%{x:,} incidents<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="Which EMS call types make up demand")
    fig.update_xaxes(showgrid=True, gridcolor=ink["grid"], title="incidents",
                     range=_headroom(d["incidents"]), **COUNT_TICKS)
    fig.update_yaxes(showgrid=False, title=None)
    _lock_category_order(fig, d["type"])
    return _style(fig, ink, height=max(340, 28 * len(d) + 110))


# ---------------------------------------------------------------------------
# Step 2 -- AHP
# ---------------------------------------------------------------------------

def chart_ahp_weights(weights: pd.Series) -> go.Figure:
    s, _, ink = tokens()
    d = weights.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d.to_numpy() * 100, y=list(d.index), orientation="h",
        marker=dict(color=s[1], line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in d], textposition="outside",
        textfont=dict(color=ink["secondary"]),
        hovertemplate="%{y}<br>weight %{x:.1f}%<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="Derived criterion weights")
    fig.update_xaxes(showgrid=True, gridcolor=ink["grid"], title="weight (%)",
                     range=[0, max(100.0, float(d.max() * 115))])
    fig.update_yaxes(showgrid=False, title=None)
    _lock_category_order(fig, d.index)
    return _style(fig, ink, height=280)


def chart_ahp_ranking(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    s, _, ink = tokens()
    d = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["ahp_score"], y=d["zip_code"].astype(str), orientation="h",
        marker=dict(color=s[1], line=dict(width=0)),
        text=[f"{v:.3f}" for v in d["ahp_score"]], textposition="outside",
        textfont=dict(color=ink["secondary"]),
        hovertemplate="ZIP %{y}<br>priority %{x:.3f}<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title=f"Top {top_n} sites by priority score")
    fig.update_xaxes(showgrid=True, gridcolor=ink["grid"], title="priority score",
                     range=_headroom(d["ahp_score"], 1.18))
    fig.update_yaxes(showgrid=False, title=None)
    _lock_category_order(fig, d["zip_code"])
    return _style(fig, ink, height=max(340, 28 * len(d) + 122))


# ---------------------------------------------------------------------------
# Step 3 -- covering
# ---------------------------------------------------------------------------

def chart_assumption_grid(df: pd.DataFrame) -> go.Figure:
    """Stations required across the travel assumptions, as a labelled grid.

    A heatmap because the reading that matters is the gradient, not any single
    cell: the eye should land on "this doubles across the grid" before it
    lands on any one number. Every cell still carries its count, since the
    corner values are the ones that get quoted.
    """
    _, _, ink = tokens()
    piv = df.pivot(index="circuity", columns="speed_kmh", values="stations_required")
    ramp = seq()
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(),
        x=[f"{c:g} km/h" for c in piv.columns],
        y=[f"×{r:g}" for r in piv.index],
        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)],
        text=piv.to_numpy(), texttemplate="%{text}",
        textfont=dict(size=13, family=FONT),
        hovertemplate="%{x} at circuity %{y}<br>"
                      "%{z} stations for full coverage<extra></extra>",
        colorbar=dict(title=dict(text="stations", font=dict(color=ink["secondary"])),
                      tickfont=dict(color=ink["secondary"]), thickness=12),
        xgap=2, ygap=2,
    ))
    fig.update_layout(title="Stations needed across the travel assumptions")
    fig.update_xaxes(title="effective speed")
    fig.update_yaxes(title="road circuity", autorange="reversed")
    return _style(fig, ink, height=320)


def chart_threshold_stations(df: pd.DataFrame) -> go.Figure:
    """Stations required at each response limit. One measure, one axis."""
    s, _, ink = tokens()
    fig = go.Figure(go.Bar(
        x=[f"{t} min" for t in df["threshold_min"]], y=df["stations_required"],
        marker=dict(color=s[1], line=dict(width=0)),
        text=df["stations_required"], textposition="outside",
        textfont=dict(color=ink["secondary"]),
        hovertemplate="%{x} limit<br>%{y} stations required<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="Stations needed for full coverage")
    # Outside labels sit above the bar, and Plotly does not extend a vertical
    # axis to fit them — without headroom the tallest bar's count is clipped.
    # Whole stations only, but a fixed dtick=1 labels every integer up to the
    # tallest bar -- at a 5-minute limit that is 19 tick labels stacked on one
    # another. Step up so roughly six labels are drawn whatever the scale.
    top = int(df["stations_required"].max()) if len(df) else 1
    fig.update_yaxes(title="stations", dtick=max(1, -(-top // 6)),
                     range=_headroom(df["stations_required"], 1.18))
    return _style(fig, ink, height=300)


def chart_threshold_coverage(df: pd.DataFrame) -> go.Figure:
    """Demand covered at each limit -- a separate chart, never a second y-axis."""
    s, _, ink = tokens()
    fig = go.Figure(go.Bar(
        x=[f"{t} min" for t in df["threshold_min"]], y=df["demand_covered_pct"],
        marker=dict(color=s[1], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in df["demand_covered_pct"]], textposition="outside",
        textfont=dict(color=ink["secondary"]),
        hovertemplate="%{x} limit<br>%{y:.1f}% of demand covered<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="Demand covered at that station count")
    fig.update_yaxes(title="% of demand", range=[0, 108])
    return _style(fig, ink, height=300)


def chart_marginal_curve(df: pd.DataFrame, mark_at: int | None = None) -> go.Figure:
    """Coverage as a function of the station budget P.

    The curve is the argument for the maximal-covering framing: the first
    handful of stations buy most of the city. ``mark_at`` is the budget the
    plan on screen actually chose, called out on the curve so the reader can
    see where that plan sits on the trade-off rather than reading it off the
    gridlines.
    """
    s, _, ink = tokens()
    fig = go.Figure(go.Scatter(
        x=df["stations"], y=df["demand_covered_pct"], mode="lines+markers",
        line=dict(color=s[1], width=2), marker=dict(size=7, color=s[1]),
        fill="tozeroy", fillcolor=_alpha(s[1], 0.10),
        hovertemplate="%{x} stations<br>%{y:.1f}% of demand covered<extra></extra>",
    ))
    if mark_at:
        row = df[df["stations"] == mark_at]
        if not row.empty:
            y = float(row["demand_covered_pct"].iloc[0])
            fig.add_trace(go.Scatter(
                x=[mark_at], y=[y], mode="markers",
                marker=dict(size=12, color=s[1], line=dict(width=2,
                                                           color=ink["surface"])),
                hovertemplate=f"{mark_at} stations — the plan on screen"
                              "<extra></extra>",
            ))
            fig.add_annotation(
                x=mark_at, y=y, text=f"this plan: <b>{mark_at}</b> → {y:.1f}%",
                showarrow=True, arrowhead=0, arrowcolor=ink["muted"],
                ax=24, ay=38, xanchor="left",
                font=dict(color=ink["secondary"], size=11),
            )
    fig.update_layout(title="Coverage vs station budget",
                      hovermode="x unified")
    fig.update_xaxes(title="number of stations (P)",
                     dtick=max(1, -(-int(df["stations"].max()) // 10)))
    fig.update_yaxes(title="% of demand covered", range=[0, 108], ticksuffix="%")
    return _style(fig, ink, height=320)


def chart_comparison(rows: list[dict]) -> go.Figure:
    """Our placement vs Seattle's real stations, at the same response limits.

    Plotted on *station count*, not on demand covered. Both plans cover
    essentially all of the demand at every limit -- 100% against 99.7%, 100%
    and 100% -- so a coverage chart draws six bars of the same height and the
    finding reads as "no difference". The difference is the cost of buying that
    coverage: 7 stations against 36 at the 8-minute limit. Coverage is still
    carried, as the label on each bar and in the table beside the chart, so the
    parity that makes the station counts comparable stays on screen.

    Two series -> legend is always present; colour means the same thing it
    means everywhere else in the app.
    """
    s, _, ink = tokens()
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for slot, name in ((1, "Optimised placement"), (2, "Seattle's real stations")):
        d = df[df["plan"] == name]
        fig.add_trace(go.Bar(
            name=name, x=[f"{t} min" for t in d["threshold_min"]],
            y=d["stations"], marker=dict(color=s[slot], line=dict(width=0)),
            text=[f"{int(n)}<br><span style='font-size:10px'>{p:.1f}% covered</span>"
                  for n, p in zip(d["stations"], d["demand_covered_pct"])],
            textposition="outside", textfont=dict(color=ink["secondary"]),
            customdata=d["demand_covered_pct"],
            hovertemplate=f"{name}<br>%{{x}} limit<br>%{{y}} stations"
                          "<br>%{customdata:.1f}% of demand covered<extra></extra>",
        ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title="Stations needed to cover the same demand",
                      barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.update_yaxes(title="stations", range=_headroom(df["stations"], 1.28))
    return _style(fig, ink, height=340, showlegend=True)


# ---------------------------------------------------------------------------
# Step 4 -- assignment
# ---------------------------------------------------------------------------

def chart_response_distribution(df: pd.DataFrame, threshold: float,
                                height: int = 300) -> go.Figure:
    """How many calls land in each minute of response time.

    The counts are binned here rather than by ``go.Histogram`` so the bins are
    whole minutes a reader can name ("3-4 min") and so the count axis can be
    given an integer tick step: a single hour yields a handful of calls per
    bin, and Plotly's automatic ticks then label a count axis 0.5, 1, 1.5 --
    half an incident is not a thing.
    """
    s, _, ink = tokens()
    d = df[df["served"]]
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="Response times under the optimised dispatch")
        fig.add_annotation(text="No served incidents to plot", showarrow=False,
                           font=dict(color=ink["muted"]), x=0.5, y=0.5,
                           xref="paper", yref="paper")
        return _style(fig, ink, height=height)

    top = max(float(d["response_min"].max()), float(threshold))
    edges = np.arange(0.0, np.ceil(top) + 1.0, 1.0)
    counts, _ = np.histogram(d["response_min"].to_numpy(), bins=edges)
    centres = edges[:-1] + 0.5
    cmax = int(counts.max()) if len(counts) else 1

    fig = go.Figure(go.Bar(
        x=centres, y=counts, width=0.92,
        marker=dict(color=s[1], line=dict(width=0)),
        customdata=np.stack([edges[:-1], edges[1:]], axis=-1),
        hovertemplate="%{customdata[0]:.0f}–%{customdata[1]:.0f} min"
                      "<br>%{y} incidents<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=3, cliponaxis=False)
    fig.add_vline(x=threshold, line_dash="dash", line_color=ink["muted"],
                  annotation_text=f"{threshold:g}-min limit",
                  annotation_position="top left",
                  annotation_font_color=ink["secondary"])
    fig.update_layout(title="Response times under the optimised dispatch")
    fig.update_xaxes(title="minutes to scene", dtick=1, tick0=0,
                     range=[0, edges[-1]])
    # Whole incidents only, and headroom so the limit annotation clears the bars.
    fig.update_yaxes(title="incidents", dtick=max(1, -(-cmax // 6)),
                     range=_headroom(counts, 1.25))
    return _style(fig, ink, height=height)


def chart_dispatch_comparison(optimised: float, greedy: float,
                              n_calls: int = 0) -> go.Figure:
    """LP optimum vs nearest-unit-first dispatch, over the same calls.

    The title has to name the workload. ``benchmark_vs_greedy`` deliberately
    scores both methods on the first ``k`` calls in arrival order rather than
    on the dispatch plan's own served set (see its docstring: comparing those
    would measure triage, not routing). That makes this total a different
    quantity from the "Total travel time" tile above it, which covers whatever
    calls the unrestricted plan actually served -- two numbers, both correct,
    that read as a contradiction if the chart just says "total travel time".
    """
    s, _, ink = tokens()
    scope = f" — same {n_calls} calls" if n_calls else ""
    fig = go.Figure(go.Bar(
        x=["Optimised", "Nearest-unit"],
        y=[optimised, greedy],
        marker=dict(color=[s[1], s[2]], line=dict(width=0)),
        text=[f"{optimised:.1f} min", f"{greedy:.1f} min"], textposition="outside",
        textfont=dict(color=ink["secondary"]),
        hovertemplate="%{x}<br>%{y:.1f} total minutes<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4, cliponaxis=False)
    fig.update_layout(title=f"Fleet travel time{scope}")
    fig.update_yaxes(title="total minutes", range=_headroom([optimised, greedy], 1.2))
    return _style(fig, ink, height=300)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

# CartoDB's free "positron"/"dark_matter" tiles now require an API key --
# unauthenticated requests still return HTTP 200, but the tile image itself is a
# watermark reading "API KEY REQUIRED". Esri's Gray Canvas needs no key, ships a
# matching dark variant, and is a desaturated basemap: the standard OpenStreetMap
# style is a finished map in its own right, and its greens, reds and road
# casings compete with the data drawn on top of it instead of carrying it.
_ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
         "World_{}_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}")
_ESRI_ATTR = 'Tiles &copy; <a href="https://www.esri.com/">Esri</a>'


def _base_map(center=(47.6062, -122.3321), zoom: int = 11) -> folium.Map:
    mode = theme()
    _s, _status, ink = tokens()
    m = folium.Map(
        location=list(center), zoom_start=zoom, control_scale=True,
        tiles=_ESRI.format("Dark" if mode == "dark" else "Light"),
        attr=_ESRI_ATTR, name="Basemap", max_zoom=16,
    )
    # Leaflet's own controls are styled for a generic map product; these bring
    # the zoom buttons, scale bar and legend into the app's own border radius,
    # border colour and surface so a map reads as another card on the page.
    m.get_root().header.add_child(folium.Element(f"""
<style>
  .leaflet-container {{ font-family: {FONT}; background: {ink["surface"]}; }}
  .leaflet-bar, .leaflet-control-layers {{
      border: 1px solid {ink["grid"]} !important;
      border-radius: 8px !important; box-shadow: none !important; }}
  /* Leaflet hard-codes a white chip and a dark glyph for the layers toggle,
     which stays a bright white square on the dark basemap next to zoom
     buttons that do follow the theme. */
  .leaflet-control-layers, .leaflet-control-layers-expanded {{
      background: {ink["surface"]} !important; color: {ink["primary"]}; }}
  .leaflet-control-layers-toggle {{ filter: {"invert(1)" if mode == "dark" else "none"}; }}
  .leaflet-control-layers label {{ color: {ink["primary"]}; }}
  .leaflet-control-layers-separator {{ border-top-color: {ink["grid"]}; }}
  .leaflet-bar a {{
      background: {ink["surface"]}; color: {ink["primary"]};
      border-bottom-color: {ink["grid"]}; }}
  .leaflet-bar a:hover {{ background: {ink["grid"]}; }}
  .leaflet-control-scale-line {{
      background: {ink["surface"]}cc; color: {ink["secondary"]};
      border: 1px solid {ink["grid"]}; border-radius: 4px; }}
  .leaflet-control-attribution {{
      background: {ink["surface"]}cc !important; color: {ink["muted"]} !important;
      font-size: 10px; }}
  .leaflet-control-attribution a {{ color: {ink["secondary"]} !important; }}
  /* The colour-ramp legend, drawn by _ramp_legend below. */
  .ems-legend {{
      position: absolute; top: 10px; right: 10px; z-index: 900;
      background: {ink["surface"]}f2; border: 1px solid {ink["grid"]};
      border-radius: 8px; padding: 7px 10px 6px; font-family: {FONT};
      box-shadow: 0 1px 2px rgba(0,0,0,.08); pointer-events: none; }}
  .ems-legend .cap {{
      font-size: 10px; font-weight: 600; letter-spacing: .04em;
      text-transform: uppercase; color: {ink["muted"]};
      display: block; margin-bottom: 5px; }}
  .ems-legend .bar {{ height: 10px; border-radius: 3px; }}
  .ems-legend .ends {{
      display: flex; justify-content: space-between; gap: 18px;
      font-size: 11px; color: {ink["secondary"]}; margin-top: 4px; }}
</style>"""))
    return m


def _fit(m: folium.Map, lats, lons, pad: float = 0.08) -> None:
    """Frame the map on what it actually plots, rather than trusting a fixed
    zoom_start against a hardcoded Seattle center.

    streamlit-folium mounts a map's iframe even while its tab panel is
    ``display:none`` (Streamlit keeps every tab's content in the DOM and only
    toggles visibility), so Leaflet can compute its initial view against a
    zero-size container. The symptom is silent and data-dependent -- the
    dispatch map was observed rendering two zoom levels out from its coded
    ``zoom_start=12`` when the Dispatch tab is opened after another tab has
    already been viewed, which is the normal way through the app. Fitting to
    explicit bounds from the real coordinates sidesteps the stale-container
    math entirely: whatever size Leaflet thinks it has, the markers end up
    framed.
    """
    lats, lons = list(lats), list(lons)
    if not lats:
        return
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    if lat_min == lat_max and lon_min == lon_max:
        # A single point has no extent to fit; hold the default city zoom
        # centred on it instead of asking Leaflet to zoom in on nothing.
        m.location = [lat_min, lon_min]
        return
    # A small fixed pad (in degrees) rather than Leaflet's own padding option,
    # so a one-station result still frames with visible context around it.
    m.fit_bounds([[lat_min - pad, lon_min - pad], [lat_max + pad, lon_max + pad]])


def _dot_icon(color: str, size: int = 18) -> folium.DivIcon:
    """A plain filled dot in the app's own token colours.

    Leaflet's default teardrop pin (folium.Icon) reads as generic map-product
    chrome next to the rest of the app's minimal circle-marker style -- this
    keeps every point marker on every map the same simple shape.
    """
    html = (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:{color};border:2px solid #ffffff;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.4);"></div>'
    )
    half = size // 2
    return folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(half, half))


def _ramp_legend(m: folium.Map, ramp, vmin: float, vmax: float, caption: str) -> None:
    """Draw the sequential ramp's key in the app's own type and tokens.

    branca's built-in legend is a d3 SVG on a fixed internal geometry: the
    caption's baseline is pinned 21px below the group origin while the ramp
    grows down from that same origin, so any ramp thick enough to read runs
    straight through the caption, and the leftmost tick label is struck through
    by the axis tick mark. Two numbers and a gradient is all this needs.
    """
    stops = ", ".join(ramp(vmin + (vmax - vmin) * i / 8.0) for i in range(9))
    m.get_root().html.add_child(folium.Element(
        f'<div class="ems-legend">'
        f'<span class="cap">{caption}</span>'
        f'<div class="bar" style="background:linear-gradient(90deg,{stops});"></div>'
        f'<div class="ends"><span>{int(vmin):,}</span><span>{int(vmax):,}</span></div>'
        f'</div>'))


def map_demand(zones: pd.DataFrame, zip_geojson: dict | None = None) -> folium.Map:
    """Where demand concentrates -- graduated circles on a single-hue ramp."""
    m = _base_map()
    if zones.empty:
        return m

    _s, _status, ink = tokens()
    vmin, vmax = float(zones["demand"].min()), float(zones["demand"].max())
    # Kept for the colour lookup below; its own legend is replaced by
    # _ramp_legend, which draws the key in the app's type rather than d3's.
    ramp = cm.LinearColormap(seq(), vmin=vmin, vmax=vmax)

    if zip_geojson:
        lookup = dict(zip(zones["zip_code"].astype(str), zones["demand"]))
        folium.GeoJson(
            zip_geojson,
            name="ZIP zones",
            style_function=lambda f: {
                "fillColor": ramp(lookup.get(str(f["properties"].get("ZCTA5")), vmin))
                if str(f["properties"].get("ZCTA5")) in lookup else "#00000000",
                "color": ink["muted"], "weight": 0.5,
                "fillOpacity": 0.55 if str(f["properties"].get("ZCTA5")) in lookup else 0.0,
            },
            tooltip=folium.GeoJsonTooltip(fields=["ZCTA5"], aliases=["ZIP:"]),
        ).add_to(m)

    dmax = max(vmax, 1.0)
    fg = folium.FeatureGroup(name="Zone demand")
    for _, r in zones.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6 + 22 * (float(r["demand"]) / dmax) ** 0.5,
            color=ink["surface"], weight=1.5, fill=True,
            fill_color=ramp(float(r["demand"])), fill_opacity=0.85,
            tooltip=f"ZIP {r['zip_code']}: {int(r['demand']):,} EMS incidents",
        ).add_to(fg)
    fg.add_to(m)

    _ramp_legend(m, ramp, vmin, vmax, "EMS incidents per zone")
    folium.LayerControl(collapsed=True, position="topleft").add_to(m)
    _fit(m, zones["latitude"], zones["longitude"])
    return m


def map_coverage(result, existing: pd.DataFrame | None = None,
                 show_radius: bool = True, speed_kmh: float = 45.0,
                 circuity: float = 1.30, turnout_min: float = 1.0) -> folium.Map:
    """Chosen stations, their reach, and which zones end up covered."""
    s, status, _ = tokens()
    m = _base_map()
    zt = result.zone_table

    # Zones: covered vs not. Status colour always accompanied by a text label.
    fg_z = folium.FeatureGroup(name="Demand zones")
    dmax = max(float(zt["demand"].max()), 1.0)
    for _, r in zt.iterrows():
        col = status["good"] if r["covered"] else status["critical"]
        label = "COVERED" if r["covered"] else "NOT COVERED"
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=5 + 18 * (float(r["demand"]) / dmax) ** 0.5,
            color="#ffffff", weight=1.2, fill=True, fill_color=col, fill_opacity=0.75,
            tooltip=(f"ZIP {r['zip_code']} — {label}<br>"
                     f"{int(r['demand']):,} incidents<br>"
                     f"nearest station: {r['nearest_station_zone']} "
                     f"({r['response_min']:.1f} min)"),
        ).add_to(fg_z)
    fg_z.add_to(m)

    # Reach radius implied by the time limit under the current travel settings.
    reach_km = max((result.threshold_min - turnout_min), 0) / 60.0 * speed_kmh / circuity

    fg_s = folium.FeatureGroup(name="Optimised stations")
    for _, r in result.chosen.iterrows():
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            tooltip=f"{r.get('station', '')} — zone {r.get('zip_code', '')}",
            icon=_dot_icon(s[1], size=20),
        ).add_to(fg_s)
        if show_radius:
            folium.Circle(
                location=[r["latitude"], r["longitude"]], radius=reach_km * 1000,
                color=s[1], weight=1.2, fill=True, fill_color=s[1], fill_opacity=0.08,
                tooltip=f"{result.threshold_min:g}-min reach (~{reach_km:.1f} km)",
            ).add_to(fg_s)
    fg_s.add_to(m)

    if existing is not None and len(existing):
        fg_e = folium.FeatureGroup(name="Seattle's real stations", show=False)
        cluster = MarkerCluster().add_to(fg_e)
        for _, r in existing.iterrows():
            folium.Marker(
                location=[r["latitude"], r["longitude"]],
                tooltip=f"{r.get('station_id', 'SFD')} — {r.get('address', '')}",
                icon=_dot_icon(s[2], size=12),
            ).add_to(cluster)
        fg_e.add_to(m)

    folium.LayerControl(collapsed=True, position="topleft").add_to(m)
    # Framed on the zone table rather than the opened stations alone, so an
    # early-budget Mode-B result (few stations, same zones) still shows the
    # uncovered zones it left out instead of cropping to just what it chose.
    _fit(m, zt["latitude"], zt["longitude"])
    return m


def map_dispatch(result, ambulances: pd.DataFrame) -> folium.Map:
    """Which unit was sent where, drawn as base -> incident lines."""
    s, status, _ = tokens()
    m = _base_map(zoom=12)

    fg_b = folium.FeatureGroup(name="Ambulance bases")
    for _, r in ambulances.iterrows():
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            tooltip=f"Unit {r.get('station', '')} — base zone {r.get('zip_code', '')}",
            icon=_dot_icon(s[1], size=18),
        ).add_to(fg_b)
    fg_b.add_to(m)

    fg_i = folium.FeatureGroup(name="Incidents & dispatch")
    for _, r in result.assignments.iterrows():
        served = bool(r["served"])
        col = status["good"] if served else status["critical"]
        tip = (f"{r['incident_number']} — {r['type']}<br>{r['address']}<br>"
               + (f"unit {r['ambulance']} · {r['response_min']:.1f} min"
                  if served else "UNSERVED — no unit available"))
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]], radius=7,
            color="#ffffff", weight=1.5, fill=True, fill_color=col,
            fill_opacity=0.9, tooltip=tip,
        ).add_to(fg_i)
        if served and not np.isnan(r["base_lat"]):
            folium.PolyLine(
                locations=[[r["base_lat"], r["base_lon"]], [r["latitude"], r["longitude"]]],
                color=s[1], weight=2, opacity=0.65,
                tooltip=f"{r['ambulance']} → {r['incident_number']} ({r['response_min']:.1f} min)",
            ).add_to(fg_i)
    fg_i.add_to(m)

    folium.LayerControl(collapsed=True, position="topleft").add_to(m)
    _fit(m, pd.concat([ambulances["latitude"], result.assignments["latitude"]]),
         pd.concat([ambulances["longitude"], result.assignments["longitude"]]))
    return m
