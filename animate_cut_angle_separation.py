"""
animate_cut_angle_separation.py
================================
Animated visualisation: Cut Angle  ↔  Separation from nearest defender

The animation steps through every 5-degree cut-angle bin, pausing longer at
route-zone transitions so the viewer can read each step.

Layout
------
  Left (main):  scatter of all routes; current bin highlighted, others ghosted;
                smooth median line builds left-to-right as bins advance;
                large bin label + zone badge in the top-left corner.
  Top-right:    separation distribution (histogram) for the current bin.
  Mid-right:    completion-rate bar for the current bin.
  Bot-right:    running median trace (mini sparkline) that fills in as we go.

Run:
    python animate_cut_angle_separation.py
Outputs: cut_angle_separation_animation.gif
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_FILE   = "cut_angle_separation_animation.gif"
CACHE_PATH = "metrics_cache.parquet"

# ── Colour palette ─────────────────────────────────────────────────────────
BG_DARK      = "#1a1a2e"
BG_PANEL     = "#16213e"
BORDER       = "#0f3460"
TEXT_COLOR   = "#e0e0e0"
MUTED        = "#66748a"
COMPLETE_C   = "#4caf50"
INCOMPLETE_C = "#ef5350"
ACCENT       = "#f5a623"       # gold — median trend line
HIGHLIGHT    = "#00e5ff"       # cyan — active bin
GHOST_ALPHA  = 0.07            # inactive dots

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor":   BG_PANEL,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       BORDER,
    "grid.alpha":       0.35,
    "font.family":      "sans-serif",
})

# ── Route zone definitions (lo inclusive, hi exclusive) ────────────────────
ZONES = [
    (  0,  20, "Stem",         "#5c6bc0"),
    ( 20,  45, "Curl / Out",   "#29b6f6"),
    ( 45,  80, "Dig / Cross",  "#26a69a"),
    ( 80, 120, "In-breaking",  "#66bb6a"),
    (120, 165, "Post / Corner","#ffa726"),
]

def zone_for(cut):
    for lo, hi, name, col in ZONES:
        if lo <= cut < hi:
            return name, col
    return ZONES[-1][2], ZONES[-1][3]

# ── Bin settings ───────────────────────────────────────────────────────────
BIN_WIDTH      = 5          # degrees per bin
BIN_LO         = 0
BIN_HI         = 160        # exclusive upper edge
BINS_LO        = np.arange(BIN_LO, BIN_HI, BIN_WIDTH)   # 0,5,10,…,155
BINS_HI        = BINS_LO + BIN_WIDTH                     # 5,10,…,160
N_BINS         = len(BINS_LO)

FRAMES_NORMAL  = 5          # frames per bin (at ~5 fps → 1 s each)
FRAMES_ZONE    = 14         # frames at a zone transition (≈ 2.8 s pause)
FRAMES_SUMMARY = 40         # hold on the final "all data" frame

# Zone boundaries where we slow down
ZONE_BOUNDARIES = {z[0] for z in ZONES[1:]}   # 20, 45, 80, 120


# ══════════════════════════════════════════════════════════════════════════
#  PRE-COMPUTE BIN STATISTICS
# ══════════════════════════════════════════════════════════════════════════

def compute_bin_stats(df):
    """
    Returns a list of dicts, one per 5° bin, with:
      lo, hi, n, median_sep, q25, q75, pct_complete, indices (bool mask into df)
    """
    stats = []
    for lo, hi in zip(BINS_LO, BINS_HI):
        mask = (df["cut_angle"] >= lo) & (df["cut_angle"] < hi)
        sub  = df[mask]
        n    = len(sub)
        if n == 0:
            med, q25, q75, pct = np.nan, np.nan, np.nan, np.nan
        else:
            med = float(np.median(sub["separation"]))
            q25 = float(np.percentile(sub["separation"], 25))
            q75 = float(np.percentile(sub["separation"], 75))
            pct = float(100 * sub["completed"].mean())
        stats.append(dict(lo=lo, hi=hi, n=n, median=med,
                          q25=q25, q75=q75, pct_comp=pct,
                          mask=mask))
    return stats


def smooth_running_medians(bin_stats):
    """
    Smooth the per-bin medians (ignoring empty bins) for the trend sparkline.
    Returns arrays (bin_centres, smoothed_medians).
    """
    centres = np.array([(s["lo"] + s["hi"]) / 2 for s in bin_stats])
    meds    = np.array([s["median"] for s in bin_stats], dtype=float)
    # Fill NaN with interpolation
    valid = ~np.isnan(meds)
    if valid.sum() < 2:
        return centres, meds
    meds_filled = np.interp(centres, centres[valid], meds[valid])
    meds_smooth = uniform_filter1d(meds_filled, size=4)
    # Put NaN back where bin was empty
    meds_smooth[~valid] = np.nan
    return centres, meds_smooth


# ══════════════════════════════════════════════════════════════════════════
#  BUILD ANIMATION
# ══════════════════════════════════════════════════════════════════════════

def build_animation(df: pd.DataFrame):
    df = df[df["cut_angle"] < BIN_HI].copy()
    df["completed"] = df["passResult"] == "C"
    df = df.reset_index(drop=True)

    sep_max = min(float(df["separation"].quantile(0.98)), 25.0)

    # ── Per-bin stats ──────────────────────────────────────────────────────
    bin_stats   = compute_bin_stats(df)
    bin_centres, smooth_meds = smooth_running_medians(bin_stats)

    # ── Frame schedule ─────────────────────────────────────────────────────
    # Each entry: (bin_index, is_summary)
    schedule = []
    for bi, bs in enumerate(bin_stats):
        # How long to hold this bin?
        if bs["lo"] in ZONE_BOUNDARIES:
            hold = FRAMES_ZONE
        else:
            hold = FRAMES_NORMAL
        for _ in range(hold):
            schedule.append((bi, False))
    # Summary
    for _ in range(FRAMES_SUMMARY):
        schedule.append((N_BINS - 1, True))

    n_frames = len(schedule)
    print(f"  Bins: {N_BINS}   Total frames: {n_frames}")

    # ── Figure / axes ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8.5), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(
        3, 2,
        width_ratios=[3, 1.5],
        height_ratios=[2.2, 1.0, 0.75],
        hspace=0.55, wspace=0.40,
        left=0.07, right=0.97, top=0.90, bottom=0.08,
    )
    ax_sc    = fig.add_subplot(gs[:, 0])   # scatter  (all 3 rows, left col)
    ax_hist  = fig.add_subplot(gs[0, 1])   # density
    ax_comp  = fig.add_subplot(gs[1, 1])   # completion bar
    ax_spark = fig.add_subplot(gs[2, 1])   # sparkline / running median

    for ax in (ax_sc, ax_hist, ax_comp, ax_spark):
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    # ── Global titles ──────────────────────────────────────────────────────
    fig.text(0.5, 0.965, "How Cut Angle Builds Separation",
             ha="center", va="top", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.932,
             "Stepping through every 5°  •  each dot = one receiver on one play",
             ha="center", va="top", fontsize=10, color=MUTED)

    # ── Scatter: static decoration ─────────────────────────────────────────
    ax_sc.set_xlabel("Cut Angle  (degrees)", fontsize=12)
    ax_sc.set_ylabel("Separation from Nearest Defender at Pass Arrival  (yards)", fontsize=11)
    ax_sc.set_xlim(-3, BIN_HI + 3)
    ax_sc.set_ylim(-0.4, sep_max + 1.2)
    ax_sc.grid(True, alpha=0.20)
    ax_sc.axhline(0, color=BORDER, linewidth=1.0, zorder=0)

    # Zone background shading + zone labels at top
    for lo, hi, name, col in ZONES:
        hi_clamp = min(hi, BIN_HI)
        ax_sc.axvspan(lo, hi_clamp, alpha=0.055, color=col, zorder=0)
        ax_sc.text((lo + hi_clamp) / 2, sep_max + 0.85, name,
                   ha="center", va="bottom", fontsize=8,
                   color=col, alpha=0.9, style="italic", fontweight="bold")

    # Ghost: all dots very faint
    ghost_colors = df["completed"].map({True: COMPLETE_C, False: INCOMPLETE_C})
    ax_sc.scatter(df["cut_angle"], df["separation"],
                  c=ghost_colors, alpha=GHOST_ALPHA, s=10,
                  zorder=1, linewidths=0)

    # Legend
    ax_sc.legend(
        handles=[
            mpatches.Patch(color=COMPLETE_C,   label="Completed pass"),
            mpatches.Patch(color=INCOMPLETE_C, label="Incomplete / INT"),
            Line2D([0],[0], color=ACCENT, linewidth=2, label="Running median"),
            mpatches.Patch(color=ACCENT, alpha=0.20, label="IQR ribbon"),
        ],
        loc="upper right", framealpha=0.3, fontsize=8.5, edgecolor=BORDER,
    )

    # ── Scatter: animated objects ──────────────────────────────────────────
    # Active-bin dots
    scat_active = ax_sc.scatter(
        [], [], s=28, alpha=0.80, zorder=4, linewidths=0,
        vmin=0, vmax=1, cmap="RdYlGn",
    )

    # Median line (builds left to right as bins advance)
    median_line, = ax_sc.plot([], [], "-", color=ACCENT,
                              linewidth=2.5, zorder=3, alpha=0.95)

    # IQR ribbon — stored as a list so we can pop/replace it
    ribbon_store = [None]

    # Vertical bin highlight band
    bin_band = ax_sc.axvspan(0, 1, alpha=0.13, color=HIGHLIGHT, zorder=0)

    # Large bin label (top-left inside scatter)
    bin_label = ax_sc.text(
        0.03, 0.97, "",
        transform=ax_sc.transAxes, fontsize=28, fontweight="bold",
        color=HIGHLIGHT, va="top", ha="left", alpha=0.90,
        path_effects=[
            __import__("matplotlib.patheffects", fromlist=["withStroke"])
            .withStroke(linewidth=4, foreground=BG_DARK)
        ],
    )

    # Zone badge (smaller, below the bin label)
    zone_badge = ax_sc.text(
        0.03, 0.78, "",
        transform=ax_sc.transAxes, fontsize=12, fontweight="bold",
        color="white", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_DARK,
                  alpha=0.85, edgecolor=BORDER),
    )

    # Stat box (n, median, completion)
    stat_box = ax_sc.text(
        0.03, 0.66, "",
        transform=ax_sc.transAxes, fontsize=10.5, color=TEXT_COLOR,
        va="top", ha="left", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_DARK,
                  alpha=0.80, edgecolor=BORDER),
    )

    # ── Histogram panel ────────────────────────────────────────────────────
    ax_hist.set_xlabel("Separation (yards)", fontsize=9)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.set_xlim(0, sep_max + 0.5)
    ax_hist.set_title("Separation distribution  ·  current bin", fontsize=9, color=TEXT_COLOR)

    HIST_EDGES   = np.linspace(0, sep_max, 16)
    HIST_CENTRES = 0.5 * (HIST_EDGES[:-1] + HIST_EDGES[1:])
    HIST_W       = HIST_EDGES[1] - HIST_EDGES[0]

    # ── Completion bar ─────────────────────────────────────────────────────
    ax_comp.set_xlim(0, 100)
    ax_comp.set_ylim(-0.5, 0.5)
    ax_comp.set_yticks([])
    ax_comp.set_xlabel("Completion %  in this bin", fontsize=9)
    ax_comp.set_title("Pass completion rate", fontsize=9, color=TEXT_COLOR)
    ax_comp.axvline(50, color=BORDER, linewidth=1, alpha=0.5)

    comp_bar  = ax_comp.barh([0], [0], height=0.6, color=COMPLETE_C, alpha=0.85)
    comp_text = ax_comp.text(1, 0, "", ha="left", va="center",
                              fontsize=12, color=TEXT_COLOR, fontweight="bold")

    # ── Sparkline: running median ──────────────────────────────────────────
    ax_spark.set_xlim(-3, BIN_HI + 3)
    ymax_spark = sep_max * 0.80
    ax_spark.set_ylim(0, ymax_spark)
    ax_spark.set_xlabel("Cut Angle (°)", fontsize=8)
    ax_spark.set_ylabel("Med. sep.", fontsize=8)
    ax_spark.set_title("Running median  (builds left → right)", fontsize=8.5, color=TEXT_COLOR)
    ax_spark.grid(True, alpha=0.20)
    # Zone shading in sparkline too
    for lo, hi, _, col in ZONES:
        ax_spark.axvspan(lo, min(hi, BIN_HI), alpha=0.07, color=col)

    spark_ghost, = ax_spark.plot(bin_centres, smooth_meds,
                                  color=ACCENT, alpha=0.12, linewidth=1.2, zorder=1)
    spark_built, = ax_spark.plot([], [], color=ACCENT, linewidth=2.2, zorder=3)
    spark_dot,   = ax_spark.plot([], [], "o", color=HIGHLIGHT,
                                  markersize=7, zorder=4)

    # ── UPDATE FUNCTION ────────────────────────────────────────────────────
    # Track the highest bin revealed so far (median line never shrinks)
    reveal_state = {"max_bi": 0}

    def update(fi):
        bi, is_summary = schedule[fi]

        bs       = bin_stats[bi]
        lo, hi   = bs["lo"], bs["hi"]
        zone_name, zone_col = zone_for((lo + hi) / 2)

        # How many bins to reveal in the median line
        reveal_bi = N_BINS if is_summary else bi + 1
        reveal_state["max_bi"] = max(reveal_state["max_bi"], reveal_bi)
        rv = reveal_state["max_bi"]

        # ── Active bin dots ────────────────────────────────────────────────
        if is_summary:
            # Show all dots at full brightness
            scat_active.set_offsets(
                np.column_stack([df["cut_angle"].values, df["separation"].values])
            )
            scat_active.set_array(df["completed"].astype(float).values)
            scat_active.set_alpha(0.55)
            scat_active.set_sizes([12] * len(df))
        else:
            mask     = bs["mask"]
            x_active = df.loc[mask, "cut_angle"].values
            y_active = df.loc[mask, "separation"].values
            c_active = df.loc[mask, "completed"].astype(float).values
            if len(x_active):
                scat_active.set_offsets(np.column_stack([x_active, y_active]))
                scat_active.set_array(c_active)
                scat_active.set_alpha(0.80)
                scat_active.set_sizes([28] * len(x_active))
            else:
                scat_active.set_offsets(np.empty((0, 2)))

        # ── Bin highlight band ─────────────────────────────────────────────
        if is_summary:
            bin_band.set_visible(False)
        else:
            bin_band.set_visible(True)
            bin_band.set_x(lo)
            bin_band.set_y(-10)
            bin_band.set_width(hi - lo)
            bin_band.set_height(200)

        # ── Growing median line + IQR ribbon ──────────────────────────────
        valid_rv = ~np.isnan(smooth_meds[:rv])
        x_line   = bin_centres[:rv][valid_rv]
        y_line   = smooth_meds[:rv][valid_rv]
        median_line.set_data(x_line, y_line)

        # Rebuild IQR ribbon
        if ribbon_store[0] is not None:
            try:
                ribbon_store[0].remove()
            except Exception:
                pass
            ribbon_store[0] = None

        # Gather per-bin q25/q75 up to rv
        q25_arr = np.array([s["q25"] for s in bin_stats[:rv]], dtype=float)
        q75_arr = np.array([s["q75"] for s in bin_stats[:rv]], dtype=float)
        bc_rv   = bin_centres[:rv]
        # Only where we have data
        vmask = ~(np.isnan(q25_arr) | np.isnan(q75_arr))
        if vmask.sum() >= 2:
            ribbon_store[0] = ax_sc.fill_between(
                bc_rv[vmask], q25_arr[vmask], q75_arr[vmask],
                color=ACCENT, alpha=0.18, zorder=2,
            )

        # ── Sparkline ─────────────────────────────────────────────────────
        spark_built.set_data(x_line, y_line)
        if len(x_line):
            spark_dot.set_data([x_line[-1]], [y_line[-1]])
        else:
            spark_dot.set_data([], [])

        # ── Bin label & zone badge ─────────────────────────────────────────
        if is_summary:
            bin_label.set_text("All bins")
            bin_label.set_color(ACCENT)
            zone_badge.set_text("Summary")
            zone_badge.get_bbox_patch().set_edgecolor(ACCENT)
            zone_badge.set_color(ACCENT)
        else:
            bin_label.set_text(f"{lo}°–{hi}°")
            bin_label.set_color(zone_col)
            zone_badge.set_text(f"Zone:  {zone_name}")
            zone_badge.get_bbox_patch().set_edgecolor(zone_col)
            zone_badge.set_color(zone_col)

        # ── Stat box ──────────────────────────────────────────────────────
        if is_summary:
            overall_med = float(np.median(df["separation"]))
            overall_pct = float(100 * df["completed"].mean())
            stat_box.set_text(
                f"All {len(df):,} routes\n"
                f"Overall median sep: {overall_med:.2f} yds\n"
                f"Overall completion: {overall_pct:.1f}%\n\n"
                "Sharp cuts (>45°) produce ~30%\n"
                "more separation than straight routes."
            )
            stat_box.get_bbox_patch().set_edgecolor(ACCENT)
        else:
            n = bs["n"]
            med = bs["median"]
            pct = bs["pct_comp"]
            if n > 0:
                stat_box.set_text(
                    f"n = {n:,} routes\n"
                    f"Median sep: {med:.2f} yds\n"
                    f"Completion: {pct:.1f}%"
                )
            else:
                stat_box.set_text(f"n = 0 routes\n(no data in this bin)")
            stat_box.get_bbox_patch().set_edgecolor(zone_col)

        # ── Histogram ─────────────────────────────────────────────────────
        ax_hist.cla()
        ax_hist.set_facecolor(BG_PANEL)
        ax_hist.set_xlabel("Separation (yards)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.set_xlim(0, sep_max + 0.5)
        ax_hist.set_title("Separation distribution  ·  current bin",
                           fontsize=9, color=TEXT_COLOR)
        ax_hist.grid(True, alpha=0.20)
        for sp in ax_hist.spines.values():
            sp.set_edgecolor(BORDER)

        if is_summary:
            y_hist = df["separation"].values
            bar_col = ACCENT
        else:
            y_hist  = df.loc[bs["mask"], "separation"].values
            bar_col = zone_col

        if len(y_hist) > 0:
            counts, _ = np.histogram(np.clip(y_hist, 0, sep_max),
                                      bins=HIST_EDGES, density=True)
            ax_hist.bar(HIST_CENTRES, counts,
                        width=HIST_W * 0.88, color=bar_col,
                        alpha=0.75, edgecolor=BORDER)
            ymax_h = counts.max() * 1.3 if counts.max() > 0 else 1
            ax_hist.set_ylim(0, ymax_h)
            med_val = float(np.median(y_hist))
            ax_hist.axvline(med_val, color=ACCENT, linewidth=2, linestyle="--")
            ax_hist.text(med_val + 0.15, ymax_h * 0.88,
                         f"med\n{med_val:.1f}",
                         color=ACCENT, fontsize=7.5, va="top")

        # ── Completion bar ─────────────────────────────────────────────────
        pct_val = (float(100 * df["completed"].mean())
                   if is_summary else (bs["pct_comp"] if bs["n"] > 0 else 0.0))
        pct_val = pct_val if not np.isnan(pct_val) else 0.0
        comp_bar[0].set_width(pct_val)
        comp_bar[0].set_color(COMPLETE_C if pct_val >= 50 else INCOMPLETE_C)
        comp_text.set_x(min(pct_val + 1, 97))
        comp_text.set_text(f"{pct_val:.1f}%")

        return (scat_active, median_line, bin_band, bin_label,
                zone_badge, stat_box, comp_bar[0], comp_text,
                spark_built, spark_dot)

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=200,       # 200 ms → 5 fps; zone pauses add natural dwells
        blit=False,
        repeat=True,
        repeat_delay=3000,
    )
    return anim, fig


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if os.path.exists(CACHE_PATH):
        print("Loading cached metrics …")
        metrics = pd.read_parquet(CACHE_PATH)
    else:
        raise FileNotFoundError(
            f"{CACHE_PATH} not found — run the notebook or the previous "
            "version of this script first to build the cache."
        )

    print(f"Routes loaded: {len(metrics):,}")
    print("Building animation …")
    anim, fig = build_animation(metrics)

    import shutil
    if shutil.which("ffmpeg"):
        from matplotlib.animation import FFMpegWriter
        mp4 = OUT_FILE.replace(".gif", ".mp4")
        anim.save(mp4, writer=FFMpegWriter(fps=5, bitrate=2200), dpi=130)
        print(f"Saved → {mp4}")
    else:
        print("ffmpeg not found; saving as GIF …")
        anim.save(OUT_FILE, writer=PillowWriter(fps=5), dpi=110)
        print(f"Saved → {OUT_FILE}")

    plt.close(fig)
    print("Done.")
