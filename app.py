"""
app.py  —  Cut Angle vs. Separation Explorer
=============================================
Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cut Angle vs. Separation",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background that matches the GIF palette */
[data-testid="stAppViewContainer"] {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #16213e;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
h1, h2, h3, h4 { color: #e0e0e0 !important; }
p, li, label { color: #cccccc !important; }

/* Metric cards */
.metric-card {
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #00e5ff;
}
.metric-card .label {
    font-size: 0.82rem;
    color: #7a8ba0;
    margin-top: 0.2rem;
}

/* Zone pill badges */
.zone-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px;
    color: #fff;
}

/* Divider */
hr { border-color: #0f3460; }

/* GIF container */
.gif-container {
    border: 1px solid #0f3460;
    border-radius: 12px;
    overflow: hidden;
    background: #16213e;
    padding: 8px;
}

/* Caption */
.caption {
    font-size: 0.78rem;
    color: #66748a;
    text-align: center;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────
CACHE = "metrics_cache.parquet"

@st.cache_data
def load_data():
    if not os.path.exists(CACHE):
        st.error(f"Cache file `{CACHE}` not found. Run `animate_cut_angle_separation.py` first.")
        st.stop()
    df = pd.read_parquet(CACHE)
    df["completed"] = df["passResult"] == "C"
    df = df[df["cut_angle"] < 160].reset_index(drop=True)
    return df

df = load_data()

# ── Zone definitions ───────────────────────────────────────────────────────
ZONES = [
    (  0,  20, "Stem",         "#5c6bc0", "Receiver runs straight — minimal direction change"),
    ( 20,  45, "Curl / Out",   "#29b6f6", "Slight breaks; defenders adjust quickly"),
    ( 45,  80, "Dig / Cross",  "#26a69a", "Decisive cuts begin to stress man coverage"),
    ( 80, 120, "In-breaking",  "#66bb6a", "Sharp angles — defenders over-commit, separation climbs"),
    (120, 160, "Post / Corner","#ffa726", "Extreme cuts yield high separation but require precise execution"),
]

def zone_for(cut):
    for lo, hi, name, col, desc in ZONES:
        if lo <= cut < hi:
            return name, col, desc
    return ZONES[-1][2], ZONES[-1][3], ZONES[-1][4]


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏈 Filter routes")
    st.markdown("---")

    pos_options = sorted(df["officialPosition"].dropna().unique())
    sel_pos = st.multiselect(
        "Receiver position",
        options=pos_options,
        default=["WR", "TE", "RB"],
    )

    cov_options = sorted(df["pff_passCoverage"].dropna().unique())
    sel_cov = st.multiselect(
        "Coverage type",
        options=cov_options,
        default=cov_options,
    )

    result_options = {"Completed (C)": "C", "Incomplete (I)": "I", "Interception (IN)": "IN"}
    sel_results = st.multiselect(
        "Pass result",
        options=list(result_options.keys()),
        default=list(result_options.keys()),
    )

    st.markdown("---")
    st.markdown("### Cut angle range (°)")
    cut_range = st.slider("", 0, 155, (0, 155), step=5)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#66748a'>"
        "Data: NFL Big Data Bowl 2022 tracking data<br>"
        "2,019 pass-route instances · Week 1–8 2021"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Filter dataframe ───────────────────────────────────────────────────────
sel_result_codes = [result_options[r] for r in sel_results]
df_f = df[
    df["officialPosition"].isin(sel_pos) &
    df["pff_passCoverage"].isin(sel_cov) &
    df["passResult"].isin(sel_result_codes) &
    (df["cut_angle"] >= cut_range[0]) &
    (df["cut_angle"] < cut_range[1] + 5)
].copy()


# ══════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='margin-bottom:0'>🏈 How Cut Angle Builds Separation</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#7a8ba0;font-size:1rem;margin-top:4px'>"
    "NFL Big Data Bowl 2022 · Weeks 1–8 · 2021 season"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════

n_routes      = len(df_f)
med_sep       = df_f["separation"].median() if n_routes else 0
pct_complete  = df_f["completed"].mean() * 100 if n_routes else 0
med_cut       = df_f["cut_angle"].median() if n_routes else 0

col1, col2, col3, col4 = st.columns(4)
for col, val, label in [
    (col1, f"{n_routes:,}",       "Routes in selection"),
    (col2, f"{med_sep:.2f} yds",  "Median separation"),
    (col3, f"{pct_complete:.1f}%","Completion rate"),
    (col4, f"{med_cut:.0f}°",     "Median cut angle"),
]:
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='value'>{val}</div>"
        f"<div class='label'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN: ANIMATION + DESCRIPTION  (two-column layout)
# ══════════════════════════════════════════════════════════════════════════

left, right = st.columns([1.15, 1], gap="large")

# ── LEFT: animated GIF ────────────────────────────────────────────────────
with left:
    st.markdown("### Animation: 5° Bin Sweep")

    gif_path = "cut_angle_separation_animation.gif"
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            gif_bytes = f.read()
        st.markdown("<div class='gif-container'>", unsafe_allow_html=True)
        st.image(gif_bytes, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='caption'>Each step = one 5° cut-angle bin &nbsp;·&nbsp; "
            "Cyan column = active bin &nbsp;·&nbsp; "
            "Gold line = running median &nbsp;·&nbsp; "
            "Longer pause at zone boundaries</p>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            f"`{gif_path}` not found. Run `animate_cut_angle_separation.py` to generate it."
        )

# ── RIGHT: description ────────────────────────────────────────────────────
with right:
    st.markdown("### How to read this chart")
    st.markdown("""
**What is cut angle?**  
Cut angle is the *maximum angular direction change* a receiver makes during their route — measured in degrees from snap to pass arrival.  A straight stem route might be 10°. A sharp out or dig is 60–90°. A full double-move reversal can reach 180°.

**What is separation?**  
Separation is the Euclidean distance (yards) between the receiver and the nearest defender at the moment the pass arrives. More separation = more time and space to make the catch.

**How the animation works**
- The scatter shows **all 2,019 route instances** from Weeks 1–8 of the 2021 NFL season.
- The chart steps through every **5° bin** (0–5°, 5–10°, … 155–160°). The active bin is highlighted in cyan; all others fade to ghosts.
- The **gold trend line** builds left-to-right as bins advance, tracing the running median separation. Watch it climb.
- The animation **pauses longer** at the five route-zone boundaries so you can absorb each transition.
- The final frame shows **all bins simultaneously** with the full IQR ribbon.

**Key takeaway**  
Routes with cuts above ~45° produce meaningfully more separation than straight or curl routes. The relationship isn't perfectly linear — very sharp cuts (>120°) are physically harder to execute at speed — but the direction is clear.
""")

    st.markdown("#### Route zones")
    for lo, hi, name, col, desc in ZONES:
        st.markdown(
            f"<span class='zone-pill' style='background:{col}'>{name}  {lo}°–{hi}°</span>"
            f"<span style='color:#aaa;font-size:0.85rem;margin-left:6px'>{desc}</span><br>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Methodology")
    st.markdown("""
| Step | Detail |
|---|---|
| **Route window** | Snap frame → pass arrival frame |
| **Cut angle** | Max `abs(dir[i] − dir[i−5])` over a 5-frame rolling window |
| **Separation** | Euclidean distance to nearest opposing player at `pass_arrived` |
| **Completion** | `passResult == "C"` from PFF scouting data |
| **Data** | 122 tracking files · 8.3 M tracking rows · 2,019 receiver routes |
""")


st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SCATTER (live, filtered)
# ══════════════════════════════════════════════════════════════════════════

st.markdown("### Interactive scatter  —  filtered selection")

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#0f3460",
    "grid.alpha":       0.4,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")

# ── Scatter ────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#16213e")
for sp in ax.spines.values():
    sp.set_edgecolor("#0f3460")

for lo, hi, name, col, _ in ZONES:
    ax.axvspan(lo, min(hi, 160), alpha=0.07, color=col)
    ax.text((lo + min(hi,160)) / 2, -0.2, name,
            ha="center", va="top", fontsize=7, color=col,
            alpha=0.85, style="italic")

if len(df_f):
    comp_mask = df_f["completed"]
    ax.scatter(df_f.loc[~comp_mask, "cut_angle"],
               df_f.loc[~comp_mask, "separation"],
               c="#ef5350", alpha=0.45, s=14, label="Incomplete / INT", linewidths=0)
    ax.scatter(df_f.loc[comp_mask, "cut_angle"],
               df_f.loc[comp_mask, "separation"],
               c="#4caf50", alpha=0.55, s=14, label="Completed", linewidths=0)

    # Smoothed median
    df_sorted = df_f.sort_values("cut_angle")
    win = max(50, int(0.15 * len(df_sorted)))
    roll_med = df_sorted["separation"].rolling(win, center=True, min_periods=15).median()
    ax.plot(df_sorted["cut_angle"], roll_med,
            color="#f5a623", linewidth=2.5, label="Smoothed median", zorder=5)

ax.set_xlabel("Cut Angle (°)", fontsize=11)
ax.set_ylabel("Separation (yards)", fontsize=11)
ax.set_title("Cut Angle vs. Separation", fontsize=12, color="#e0e0e0")
ax.set_xlim(-3, 163)
sep_max_plot = min(float(df["separation"].quantile(0.98)), 25)
ax.set_ylim(-0.4, sep_max_plot + 0.5)
ax.grid(True, alpha=0.22)
ax.legend(fontsize=9, framealpha=0.3, edgecolor="#0f3460")

# ── Completion rate by zone ────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#16213e")
for sp in ax2.spines.values():
    sp.set_edgecolor("#0f3460")

zone_names, zone_pcts, zone_ns, zone_cols = [], [], [], []
for lo, hi, name, col, _ in ZONES:
    sub = df_f[(df_f["cut_angle"] >= lo) & (df_f["cut_angle"] < hi)]
    zone_names.append(f"{name}\n({lo}°–{hi}°)")
    zone_pcts.append(sub["completed"].mean() * 100 if len(sub) else 0)
    zone_ns.append(len(sub))
    zone_cols.append(col)

bars = ax2.bar(zone_names, zone_pcts, color=zone_cols, alpha=0.80,
               edgecolor="#0f3460", linewidth=0.8)
ax2.axhline(50, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.4)
ax2.set_ylim(0, 100)
ax2.set_ylabel("Completion %", fontsize=11)
ax2.set_title("Completion Rate by Zone", fontsize=12, color="#e0e0e0")
ax2.grid(axis="y", alpha=0.22)
ax2.tick_params(axis="x", labelsize=8)

for bar, pct, n in zip(bars, zone_pcts, zone_ns):
    if n > 0:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 pct + 1.5, f"{pct:.1f}%\n(n={n})",
                 ha="center", va="bottom", fontsize=8, color="#e0e0e0")

plt.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  PER-BIN TABLE
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### Per-5° bin statistics")

rows = []
for lo in range(0, 160, 5):
    hi   = lo + 5
    sub  = df_f[(df_f["cut_angle"] >= lo) & (df_f["cut_angle"] < hi)]
    n    = len(sub)
    name, col, _ = zone_for((lo + hi) / 2)
    rows.append({
        "Bin": f"{lo}°–{hi}°",
        "Zone": name,
        "Routes (n)": n,
        "Median sep (yds)": round(sub["separation"].median(), 2) if n else "—",
        "Mean sep (yds)":   round(sub["separation"].mean(),   2) if n else "—",
        "IQR (yds)": (
            f"{sub['separation'].quantile(0.25):.2f} – "
            f"{sub['separation'].quantile(0.75):.2f}"
        ) if n else "—",
        "Completion %": f"{sub['completed'].mean()*100:.1f}%" if n else "—",
    })

tbl = pd.DataFrame(rows)

# Colour rows by zone
def colour_zone(row):
    zone_colours = {z[2]: z[3] for z in ZONES}
    c = zone_colours.get(row["Zone"], "#1a1a2e")
    return [f"background-color:{c}22; color:#e0e0e0"] * len(row)

st.dataframe(
    tbl.style.apply(colour_zone, axis=1),
    use_container_width=True,
    hide_index=True,
    height=560,
)

st.markdown(
    "<p class='caption' style='margin-top:8px'>"
    "Table reflects sidebar filters. Scroll to see all 32 bins.</p>",
    unsafe_allow_html=True,
)
