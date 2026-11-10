"""
animate_field_view.py
======================
Top-down football field animation showing THREE plays back-to-back:
  1. Low cut angle  (~12°)  — Rob Gronkowski stem route
  2. Medium cut angle (~72°) — Antonio Brown dig/cross
  3. High cut angle (~180°)  — Hayden Hurst sharp out/back

For each play the animation shows:
  • A proper NFL field (green, yard lines, hash marks, end zones)
  • The receiver (red circle + jersey number) running their route with a fading trail
  • The nearest coverage defender (blue circle + jersey number)
  • The quarterback (white dot) standing in the pocket
  • All other players (faint grey)
  • A live "cut angle arc" that appears at the moment of the sharpest cut
  • A live separation ruler drawn between receiver and nearest defender
  • A score panel showing current cut angle, separation, and frame
  • A title card between plays explaining what just happened

Output: field_animation.gif  (~3-4 MB)
"""

import glob, os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch, Circle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════

BASE      = "data"
OUT_FILE  = "field_animation.gif"
FPS       = 8          # GIF frames per second
HOLD_CARD = 28         # frames to show the title card between plays
HOLD_END  = 30         # frames to hold the final frame of each play

# Field dimensions (NFL tracking coords: x = 0-120 yd line to line, y = 0-53.3)
FIELD_X = (0, 120)
FIELD_Y = (0, 53.3)

# Colour palette
BG_DARK      = "#1a1a2e"
BORDER       = "#0f3460"
FIELD_GREEN  = "#2d5a27"
FIELD_LIGHT  = "#336b2c"
LINE_WHITE   = "#e8e8e8"
END_ZONE_L   = "#1a3a1a"
END_ZONE_R   = "#1a3a1a"
REC_COLOR    = "#e94560"      # receiver — red
DEF_COLOR    = "#0096ff"      # nearest defender — blue
QB_COLOR     = "#ffffff"      # QB — white
OTHER_OFF    = "#ff9966"      # other offense
OTHER_DEF    = "#6699cc"      # other defense
FOOTBALL_C   = "#c8820f"      # ball
TRAIL_COLOR  = "#ffcc00"      # receiver trail
SEP_COLOR    = "#00e5ff"      # separation ruler
ARC_COLOR    = "#ffcc00"      # cut angle arc
TEXT_BG      = "#0a0a1a"


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def abs_angular_diff(a, b):
    diff = (a - b + 180) % 360 - 180
    return abs(diff)

def compute_cut_angle_series(dir_series, window=5):
    """Return a per-frame cut angle (angle change over last `window` frames)."""
    dirs   = dir_series.values
    angles = np.zeros(len(dirs))
    for i in range(window, len(dirs)):
        angles[i] = abs_angular_diff(dirs[i], dirs[i - window])
    return angles

def load_play(game_id, play_id, rec_nfl_id):
    """Load all tracking rows for one play and return structured dicts."""
    fname = f"{BASE}/tracking/tracking_{game_id}.csv"
    tracking = pd.read_csv(fname)
    play     = tracking[tracking["playId"] == play_id].copy()

    pff     = pd.read_csv(f"{BASE}/pffScoutingData.csv")
    players = pd.read_csv(f"{BASE}/players.csv")

    roles = pff[(pff["gameId"] == int(game_id)) & (pff["playId"] == play_id)][
        ["nflId", "pff_role"]
    ]

    # Identify teams from receiver
    rec_team = play[play["nflId"] == rec_nfl_id]["team"].iloc[0]

    # Key frame boundaries
    snap_frame   = int(play[play["event"] == "ball_snap"]["frameId"].iloc[0])
    arrive_rows  = play[play["event"].isin(["pass_outcome_caught",
                                             "pass_outcome_incomplete",
                                             "pass_outcome_interception",
                                             "pass_arrived"])]
    arrive_frame = int(arrive_rows["frameId"].iloc[0]) if not arrive_rows.empty else play["frameId"].max()

    # Receiver trajectory (snap → arrive)
    rec_track = play[
        (play["nflId"] == rec_nfl_id) &
        (play["frameId"] >= snap_frame) &
        (play["frameId"] <= arrive_frame)
    ].sort_values("frameId").reset_index(drop=True)

    # Find nearest defender at each frame
    def_team = play[play["team"] != rec_team]["team"].unique()
    def_team = [t for t in def_team if t != "football"]

    nearest_def_id = None
    min_avg_dist   = np.inf
    for did in play["nflId"].dropna().unique():
        drow = play[(play["nflId"] == did) & (play["team"].isin(def_team))]
        if drow.empty:
            continue
        at_arrive = drow[drow["frameId"] == arrive_frame]
        if at_arrive.empty:
            continue
        rx = rec_track[rec_track["frameId"] == arrive_frame]["x"].values
        ry = rec_track[rec_track["frameId"] == arrive_frame]["y"].values
        if len(rx) == 0:
            continue
        dist = np.hypot(at_arrive["x"].values[0] - rx[0],
                        at_arrive["y"].values[0] - ry[0])
        if dist < min_avg_dist:
            min_avg_dist   = dist
            nearest_def_id = did

    def_track = play[
        (play["nflId"] == nearest_def_id) &
        (play["frameId"] >= snap_frame) &
        (play["frameId"] <= arrive_frame)
    ].sort_values("frameId").reset_index(drop=True)

    # QB
    qb_role = roles[roles["pff_role"] == "Pass"]
    qb_id   = qb_role["nflId"].values[0] if not qb_role.empty else None

    # Ball
    ball_track = play[
        (play["team"] == "football") &
        (play["frameId"] >= snap_frame) &
        (play["frameId"] <= arrive_frame)
    ].sort_values("frameId").reset_index(drop=True)

    # All other players at each frame
    others = play[
        (play["nflId"].notna()) &
        (play["nflId"] != rec_nfl_id) &
        (play["nflId"] != nearest_def_id) &
        (play["nflId"] != qb_id) &
        (play["frameId"] >= snap_frame) &
        (play["frameId"] <= arrive_frame)
    ].copy()

    # Pre-compute cut angle series for receiver
    cut_series = compute_cut_angle_series(rec_track["dir"], window=5)
    peak_cut_frame_idx = int(np.argmax(cut_series))
    peak_cut_val       = float(cut_series[peak_cut_frame_idx])

    # Player name & jersey
    rec_info = players[players["nflId"] == rec_nfl_id].iloc[0]
    def_info = players[players["nflId"] == nearest_def_id].iloc[0] \
               if nearest_def_id is not None else None

    return {
        "game_id":          game_id,
        "play_id":          play_id,
        "rec_track":        rec_track,
        "def_track":        def_track,
        "ball_track":       ball_track,
        "others":           others,
        "qb_id":            qb_id,
        "rec_team":         rec_team,
        "snap_frame":       snap_frame,
        "arrive_frame":     arrive_frame,
        "frames":           sorted(rec_track["frameId"].unique()),
        "cut_series":       cut_series,
        "peak_cut_idx":     peak_cut_frame_idx,
        "peak_cut_val":     peak_cut_val,
        "rec_name":         rec_info["displayName"],
        "rec_jersey":       str(int(play[play["nflId"]==rec_nfl_id]["jerseyNumber"].iloc[0])),
        "def_name":         def_info["displayName"] if def_info is not None else "Defender",
        "def_jersey":       str(int(play[play["nflId"]==nearest_def_id]["jerseyNumber"].iloc[0]))
                            if nearest_def_id is not None else "?",
        "play_df":          play,
    }


# ══════════════════════════════════════════════════════════════════════════
#  FIELD DRAWING
# ══════════════════════════════════════════════════════════════════════════

def draw_field(ax):
    """Draw a top-down NFL field on ax. x=0-120 (10 yd end zones), y=0-53.3."""
    ax.set_facecolor(FIELD_GREEN)
    ax.set_xlim(FIELD_X)
    ax.set_ylim(FIELD_Y)
    ax.set_aspect("equal")
    ax.axis("off")

    # End zones
    for x0, x1 in [(0, 10), (110, 120)]:
        ax.add_patch(mpatches.Rectangle((x0, 0), x1-x0, 53.3,
                                         facecolor=END_ZONE_L, zorder=1))

    # Alternating lighter stripes every 5 yards (10–110)
    for yard in range(10, 111, 10):
        ax.add_patch(mpatches.Rectangle((yard, 0), 5, 53.3,
                                         facecolor=FIELD_LIGHT, alpha=0.4, zorder=1))

    # Yard lines every 5 yards
    for x in range(10, 111, 5):
        ax.plot([x, x], [0, 53.3], color=LINE_WHITE, linewidth=0.6, alpha=0.5, zorder=2)

    # Sidelines + end-lines
    for y in [0, 53.3]:
        ax.plot([0, 120], [y, y], color=LINE_WHITE, linewidth=1.5, zorder=2)
    for x in [0, 10, 110, 120]:
        ax.plot([x, x], [0, 53.3], color=LINE_WHITE, linewidth=1.5, zorder=2)

    # Hash marks (NFL: 70'9" from each sideline ≈ 23.58 yds)
    hash_y = [23.58, 29.72]
    for x in range(10, 111, 1):
        for hy in hash_y:
            ax.plot([x, x+0.5], [hy, hy], color=LINE_WHITE, linewidth=0.6, alpha=0.6, zorder=2)

    # Yard number labels (10,20,…,50,…,40,…,10 mirrored)
    yard_nums = list(range(10, 60, 10)) + list(range(40, 0, -10))
    for i, num in enumerate(yard_nums):
        xpos = 20 + i * 10
        for ypos in [5, 48.3]:
            ax.text(xpos, ypos, str(num),
                    ha="center", va="center", fontsize=7,
                    color=LINE_WHITE, alpha=0.7, zorder=3,
                    fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════
#  TITLE CARD CONTENT
# ══════════════════════════════════════════════════════════════════════════

TITLE_CARDS = [
    # (shown BEFORE play i)
    {
        "headline": "Play 1 of 3  —  Shallow Route  (~12° cut)",
        "sub":      "Rob Gronkowski runs a simple stem route.\n"
                    "Minimal direction change → defender stays close.",
        "color":    "#5c6bc0",
    },
    {
        "headline": "Play 2 of 3  —  Decisive Cut  (~72° cut)",
        "sub":      "Antonio Brown makes a sharp in-breaking cut.\n"
                    "Defender over-commits → modest separation opens up.",
        "color":    "#26a69a",
    },
    {
        "headline": "Play 3 of 3  —  Extreme Cut  (~180° cut)",
        "sub":      "Hayden Hurst reverses direction almost completely.\n"
                    "Watch how the cut angle arc snaps open.",
        "color":    "#ffa726",
    },
]

SUMMARY_CARD = {
    "headline": "Sharp cuts create open receivers",
    "sub":      "12° cut  →  low separation\n"
                "72° cut  →  moderate separation\n"
                "180° cut  →  high separation\n\n"
                "The angle of the cut is one of the strongest\n"
                "predictors of yards of open space.",
    "color":    "#66bb6a",
}


# ══════════════════════════════════════════════════════════════════════════
#  MAIN BUILD
# ══════════════════════════════════════════════════════════════════════════

def build_animation(plays_data):
    """
    plays_data: list of dicts from load_play()
    """
    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)

    # Main field axis (left 70%) + side panel (right 30%)
    ax_field = fig.add_axes([0.01, 0.08, 0.66, 0.84])
    ax_panel = fig.add_axes([0.70, 0.08, 0.28, 0.84])

    ax_panel.set_facecolor(BG_DARK)
    for sp in ax_panel.spines.values():
        sp.set_visible(False)
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])

    draw_field(ax_field)

    # ── Build the full frame list ───────────────────────────────────────────
    # Format: ("card", card_dict) | ("play", play_dict, frame_idx) | ("hold", play_dict) | ("summary",)
    frame_list = []
    for i, pd_ in enumerate(plays_data):
        # title card before each play
        for _ in range(HOLD_CARD):
            frame_list.append(("card", TITLE_CARDS[i]))
        # actual play frames
        for fi, fid in enumerate(pd_["frames"]):
            frame_list.append(("play", pd_, fi))
        # hold on last frame
        for _ in range(HOLD_END):
            frame_list.append(("hold", pd_))

    # summary card at end
    for _ in range(HOLD_CARD + 20):
        frame_list.append(("summary",))

    n_frames = len(frame_list)
    print(f"  Total animation frames: {n_frames}")

    # ── Persistent drawing objects (we'll update them each frame) ──────────
    # Field objects
    rec_circle  = Circle((0,0), 1.0, color=REC_COLOR,  zorder=10)
    def_circle  = Circle((0,0), 1.0, color=DEF_COLOR,  zorder=9)
    qb_circle   = Circle((0,0), 0.8, color=QB_COLOR,   zorder=8, alpha=0.9)
    ball_circle = Circle((0,0), 0.5, color=FOOTBALL_C, zorder=11)

    for c in (rec_circle, def_circle, qb_circle, ball_circle):
        ax_field.add_patch(c)
        c.set_visible(False)

    # Jersey number labels
    rec_label  = ax_field.text(0, 0, "", ha="center", va="center",
                                fontsize=6.5, fontweight="bold",
                                color="white", zorder=12)
    def_label  = ax_field.text(0, 0, "", ha="center", va="center",
                                fontsize=6.5, fontweight="bold",
                                color="white", zorder=12)

    # Route trail (line)
    trail_line, = ax_field.plot([], [], "-", color=TRAIL_COLOR,
                                linewidth=2.2, alpha=0.85, zorder=7)
    def_trail,  = ax_field.plot([], [], "-", color=DEF_COLOR,
                                linewidth=1.4, alpha=0.4, zorder=6)

    # Separation ruler
    sep_line,  = ax_field.plot([], [], "-", color=SEP_COLOR,
                                linewidth=2.0, zorder=8, alpha=0.9,
                                linestyle=(0, (4,2)))
    sep_text   = ax_field.text(0, 0, "", ha="center", va="bottom",
                                fontsize=9, color=SEP_COLOR, zorder=13,
                                fontweight="bold",
                                path_effects=[pe.withStroke(linewidth=2, foreground=TEXT_BG)])

    # Direction arrow for receiver
    dir_arrow  = ax_field.annotate(
        "", xy=(0,0), xytext=(0,0),
        arrowprops=dict(arrowstyle="->", color=REC_COLOR, lw=2.0),
        zorder=11
    )

    # Cut angle arc (drawn on peak cut frame)
    cut_arc    = Arc((0,0), 4, 4, angle=0, theta1=0, theta2=0,
                     color=ARC_COLOR, linewidth=2.5, zorder=12)
    cut_arc.set_visible(False)
    ax_field.add_patch(cut_arc)
    cut_text   = ax_field.text(0, 0, "", ha="center", va="center",
                                fontsize=10, fontweight="bold",
                                color=ARC_COLOR, zorder=14,
                                path_effects=[pe.withStroke(linewidth=2.5, foreground=TEXT_BG)])

    # Other players scatter
    others_scat = ax_field.scatter([], [], s=90, zorder=5, linewidths=0)

    # ── Panel objects ──────────────────────────────────────────────────────
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)

    panel_title = ax_panel.text(
        0.5, 0.97, "", ha="center", va="top", fontsize=13, fontweight="bold",
        color="white", wrap=True, transform=ax_panel.transAxes
    )
    panel_sub = ax_panel.text(
        0.5, 0.80, "", ha="center", va="top", fontsize=10,
        color="#cccccc", transform=ax_panel.transAxes, linespacing=1.6
    )
    panel_divider = ax_panel.axhline(0.68, color=BORDER, linewidth=1, alpha=0)

    # Live stats
    stat_labels = {}
    stat_vals   = {}
    stat_items  = [
        ("cut_angle", "Cut Angle",    "°",  0.60),
        ("separation","Separation",   " yds",0.46),
        ("speed",     "Rec Speed",    " yds/s",0.32),
        ("frame",     "Frame",        "",   0.18),
    ]
    for key, lbl, unit, ypos in stat_items:
        stat_labels[key] = ax_panel.text(
            0.08, ypos, lbl + ":", ha="left", va="center",
            fontsize=10, color="#aaaaaa", transform=ax_panel.transAxes
        )
        stat_vals[key] = ax_panel.text(
            0.92, ypos, "—", ha="right", va="center",
            fontsize=12, fontweight="bold", color="white",
            transform=ax_panel.transAxes
        )

    # Player name labels in panel
    rec_name_txt = ax_panel.text(
        0.5, 0.88, "", ha="center", va="top", fontsize=9,
        color=REC_COLOR, transform=ax_panel.transAxes
    )
    def_name_txt = ax_panel.text(
        0.5, 0.84, "", ha="center", va="top", fontsize=9,
        color=DEF_COLOR, transform=ax_panel.transAxes
    )

    # Progress bar for play
    prog_bg  = ax_panel.add_patch(
        mpatches.Rectangle((0.08, 0.08), 0.84, 0.04,
                            transform=ax_panel.transAxes,
                            facecolor=BORDER, zorder=1)
    )
    prog_bar = ax_panel.add_patch(
        mpatches.Rectangle((0.08, 0.08), 0.0, 0.04,
                            transform=ax_panel.transAxes,
                            facecolor=REC_COLOR, zorder=2)
    )
    prog_label = ax_panel.text(
        0.5, 0.055, "route progress", ha="center", va="top",
        fontsize=8, color="#888888", transform=ax_panel.transAxes
    )

    # Colour bar accent strip on panel left edge
    accent_bar = ax_panel.add_patch(
        mpatches.Rectangle((0, 0), 0.03, 1,
                            transform=ax_panel.transAxes,
                            facecolor="#5c6bc0", zorder=0)
    )

    # Card overlay (shown during title/summary)
    card_patch = ax_field.add_patch(
        mpatches.FancyBboxPatch(
            (5, 8), 110, 38,
            boxstyle="round,pad=1",
            facecolor=TEXT_BG, edgecolor="#aaaaaa",
            linewidth=1.5, zorder=20, alpha=0.0
        )
    )
    card_headline = ax_field.text(
        60, 36, "", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white", zorder=21,
        path_effects=[pe.withStroke(linewidth=3, foreground=TEXT_BG)]
    )
    card_sub = ax_field.text(
        60, 24, "", ha="center", va="center",
        fontsize=11, color="#cccccc", zorder=21, linespacing=1.7
    )
    card_color_bar = ax_field.add_patch(
        mpatches.Rectangle((5, 42), 110, 2.5,
                            facecolor="#5c6bc0", zorder=22, alpha=0.0)
    )

    # Field title (bottom)
    field_title = ax_field.text(
        60, 1.2, "", ha="center", va="bottom",
        fontsize=10, color="#cccccc", zorder=5, style="italic"
    )

    # ── State helpers ──────────────────────────────────────────────────────
    def hide_field_objects():
        rec_circle.set_visible(False)
        def_circle.set_visible(False)
        qb_circle.set_visible(False)
        ball_circle.set_visible(False)
        trail_line.set_data([], [])
        def_trail.set_data([], [])
        sep_line.set_data([], [])
        sep_text.set_text("")
        cut_arc.set_visible(False)
        cut_text.set_text("")
        rec_label.set_text("")
        def_label.set_text("")
        others_scat.set_offsets(np.empty((0, 2)))
        dir_arrow.xy       = (0, 0)
        dir_arrow.xytext   = (0, 0)
        dir_arrow.set_visible(False)
        field_title.set_text("")

    def show_card(card_dict):
        hide_field_objects()
        card_patch.set_alpha(0.92)
        card_headline.set_text(card_dict["headline"])
        card_sub.set_text(card_dict["sub"])
        card_color_bar.set_facecolor(card_dict["color"])
        card_color_bar.set_alpha(1.0)
        accent_bar.set_facecolor(card_dict["color"])
        # Clear panel
        panel_title.set_text(card_dict["headline"])
        panel_title.set_color(card_dict["color"])
        panel_sub.set_text("")
        rec_name_txt.set_text("")
        def_name_txt.set_text("")
        for k in stat_vals:
            stat_vals[k].set_text("—")
        prog_bar.set_width(0)

    def render_play_frame(pd_, fi):
        """Render one frame of a play onto the field."""
        card_patch.set_alpha(0)
        card_headline.set_text("")
        card_sub.set_text("")
        card_color_bar.set_alpha(0)

        frames  = pd_["frames"]
        fid     = frames[fi]
        n_frames_play = len(frames)
        progress = (fi + 1) / n_frames_play

        rec_t  = pd_["rec_track"]
        def_t  = pd_["def_track"]
        ball_t = pd_["ball_track"]

        rec_row  = rec_t[rec_t["frameId"] == fid].iloc[0]
        def_rows = def_t[def_t["frameId"] == fid]
        ball_rows= ball_t[ball_t["frameId"] == fid]

        rx, ry = float(rec_row["x"]), float(rec_row["y"])
        rdir   = float(rec_row["dir"])
        rspd   = float(rec_row["s"])

        # Receiver circle
        rec_circle.set_center((rx, ry))
        rec_circle.set_visible(True)
        rec_label.set_position((rx, ry))
        rec_label.set_text(pd_["rec_jersey"])

        # Defender
        if not def_rows.empty:
            drow = def_rows.iloc[0]
            dx, dy = float(drow["x"]), float(drow["y"])
            def_circle.set_center((dx, dy))
            def_circle.set_visible(True)
            def_label.set_position((dx, dy))
            def_label.set_text(pd_["def_jersey"])

            # Separation ruler
            sep_dist = np.hypot(rx - dx, ry - dy)
            sep_line.set_data([rx, dx], [ry, dy])
            mx, my = 0.5*(rx+dx), 0.5*(ry+dy)
            sep_text.set_position((mx, my + 0.9))
            sep_text.set_text(f"{sep_dist:.1f} yds")

            # Defender trail
            def_trail_rows = def_t[def_t["frameId"] <= fid]
            def_trail.set_data(def_trail_rows["x"].values,
                               def_trail_rows["y"].values)
        else:
            def_circle.set_visible(False)
            sep_line.set_data([], [])
            sep_text.set_text("")
            def_trail.set_data([], [])
            sep_dist = 0.0

        # Ball
        if not ball_rows.empty:
            brow = ball_rows.iloc[0]
            ball_circle.set_center((float(brow["x"]), float(brow["y"])))
            ball_circle.set_visible(True)
        else:
            ball_circle.set_visible(False)

        # QB  (show at QB's current position)
        qb_rows = pd_["play_df"][
            (pd_["play_df"]["nflId"] == pd_["qb_id"]) &
            (pd_["play_df"]["frameId"] == fid)
        ]
        if not qb_rows.empty:
            qr = qb_rows.iloc[0]
            qb_circle.set_center((float(qr["x"]), float(qr["y"])))
            qb_circle.set_visible(True)
        else:
            qb_circle.set_visible(False)

        # Receiver trail
        trail_rows = rec_t[rec_t["frameId"] <= fid]
        trail_line.set_data(trail_rows["x"].values, trail_rows["y"].values)

        # Direction arrow (1.8 yd ahead)
        rad = np.radians(90 - rdir)   # NFL dir: 0=up(north), 90=right(east)
        arr_len = max(rspd * 0.35, 1.0)
        ax_tip  = rx + arr_len * np.cos(rad)
        ay_tip  = ry + arr_len * np.sin(rad)
        dir_arrow.xy     = (ax_tip, ay_tip)
        dir_arrow.xytext = (rx, ry)
        dir_arrow.set_visible(True)

        # Cut angle arc — show from peak cut frame onward
        cut_val = pd_["cut_series"][fi]
        if fi >= pd_["peak_cut_idx"] and pd_["peak_cut_val"] > 10:
            peak_fi  = pd_["peak_cut_idx"]
            peak_row = rec_t.iloc[peak_fi]
            px_cut   = float(peak_row["x"])
            py_cut   = float(peak_row["y"])

            # Draw the arc centred on the peak cut position
            pre_dir  = float(rec_t.iloc[max(0, peak_fi-5)]["dir"])
            post_dir = float(peak_row["dir"])

            # Convert NFL dir (0=N, CW) → math angle (0=E, CCW)
            def nfl_to_math(d):
                return (90 - d) % 360

            theta1 = min(nfl_to_math(pre_dir), nfl_to_math(post_dir))
            theta2 = max(nfl_to_math(pre_dir), nfl_to_math(post_dir))
            if theta2 - theta1 > 180:
                theta1, theta2 = theta2, theta2 + (360 - (theta2 - theta1))

            cut_arc.center = (px_cut, py_cut)
            cut_arc.width  = 5.5
            cut_arc.height = 5.5
            cut_arc.theta1 = theta1
            cut_arc.theta2 = theta2
            cut_arc.set_visible(True)
            cut_text.set_position((px_cut, py_cut + 3.5))
            cut_text.set_text(f"{pd_['peak_cut_val']:.0f}°")
        else:
            cut_arc.set_visible(False)
            cut_text.set_text("")

        # Other players
        others_frame = pd_["others"][pd_["others"]["frameId"] == fid]
        if not others_frame.empty:
            ox = others_frame["x"].values
            oy = others_frame["y"].values
            oteam = others_frame["team"].values
            oc = [OTHER_OFF if t == pd_["rec_team"] else OTHER_DEF for t in oteam]
            others_scat.set_offsets(np.column_stack([ox, oy]))
            others_scat.set_color(oc)
            others_scat.set_alpha(0.45)
        else:
            others_scat.set_offsets(np.empty((0, 2)))

        # Field title
        pct_complete = "✔ COMPLETE" if pd_["frames"][fi] in [
            pd_["frames"][-1]] and False else ""
        field_title.set_text(
            f"{pd_['rec_name']}  vs  {pd_['def_name']}  |  "
            f"Frame {fi+1}/{len(frames)}"
        )

        # Panel stats
        panel_title.set_text(f"Cut angle: {pd_['peak_cut_val']:.0f}°")
        panel_title.set_color(ARC_COLOR if fi >= pd_["peak_cut_idx"] else "white")
        rec_name_txt.set_text(f"● {pd_['rec_name']} ({pd_['rec_jersey']})")
        def_name_txt.set_text(f"● {pd_['def_name']} ({pd_['def_jersey']})")

        stat_vals["cut_angle"].set_text(
            f"{cut_val:.0f}°" if fi >= 2 else "—"
        )
        stat_vals["separation"].set_text(f"{sep_dist:.1f}")
        stat_vals["speed"].set_text(f"{rspd:.1f}")
        stat_vals["frame"].set_text(f"{fi+1} / {len(frames)}")

        prog_bar.set_width(0.84 * progress)
        # Colour accent
        accent_bar.set_facecolor(
            ZONE_COLOR_FOR_CUT(pd_["peak_cut_val"])
        )

    def ZONE_COLOR_FOR_CUT(cut):
        if cut < 20:   return "#5c6bc0"
        if cut < 45:   return "#29b6f6"
        if cut < 80:   return "#26a69a"
        if cut < 120:  return "#66bb6a"
        return "#ffa726"

    # ── Update ──────────────────────────────────────────────────────────────
    def update(fi):
        item = frame_list[fi]
        kind = item[0]

        if kind == "card":
            show_card(item[1])

        elif kind == "summary":
            show_card(SUMMARY_CARD)

        elif kind == "play":
            _, pd_, frame_fi = item
            render_play_frame(pd_, frame_fi)

        elif kind == "hold":
            _, pd_ = item
            render_play_frame(pd_, len(pd_["frames"]) - 1)

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // FPS,
        blit=False,
        repeat=True,
        repeat_delay=2000,
    )
    return anim, fig


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Three plays, chosen for their cut angle spread:
    # 1. Low   cut (~12°)  — Rob Gronkowski  game 2021090900 play 2008
    # 2. Medium cut (~72°) — Antonio Brown   game 2021090900 play 410
    # 3. High  cut (~180°) — Hayden Hurst    game 2021091910 play 1621
    PLAY_SPECS = [
        ("2021090900", 2008, 35481),   # Gronkowski low cut
        ("2021090900",  410, 35634),   # Antonio Brown mid cut
        ("2021091910", 1621, 46094),   # Hayden Hurst high cut
    ]

    plays_data = []
    for game_id, play_id, rec_id in PLAY_SPECS:
        print(f"Loading play {game_id}/{play_id} (rec {rec_id}) …")
        pd_ = load_play(game_id, play_id, rec_id)
        print(f"  → {pd_['rec_name']}  cut={pd_['peak_cut_val']:.1f}°  "
              f"frames={len(pd_['frames'])}")
        plays_data.append(pd_)

    print("Building animation …")
    anim, fig = build_animation(plays_data)

    print(f"Saving → {OUT_FILE} …")
    anim.save(OUT_FILE, writer=PillowWriter(fps=FPS), dpi=120)
    plt.close(fig)
    print("Done.")
