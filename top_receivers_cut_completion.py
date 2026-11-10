"""
Top Receivers: Most Common Cut Angle Bucket vs Completion Rate at That Angle
============================================================================
Loads the existing tracking / play data, re-runs the route-metric pipeline,
then produces a bar chart showing each top receiver's modal cut-angle bucket
(x-axis) alongside their completion rate when running that cut (bar fill).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os

# ── Matplotlib style (matches the existing notebook) ──────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#0f3460',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#e0e0e0',
    'ytick.color':      '#e0e0e0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#0f3460',
    'grid.alpha':       0.5,
    'font.family':      'sans-serif',
})

ACCENT_COLOR   = '#f5a623'   # gold
COMPLETE_COLOR = '#4caf50'   # green
INCOMPLETE_COLOR = '#ef5350' # red

# ── 1. Load Data ───────────────────────────────────────────────────────────
BASE = 'data'

plays   = pd.read_csv(f'{BASE}/plays.csv')
players = pd.read_csv(f'{BASE}/players.csv')
pff     = pd.read_csv(f'{BASE}/pffScoutingData.csv')

tracking_files = sorted(glob.glob(f'{BASE}/tracking/tracking_*.csv'))
print(f'Loading {len(tracking_files)} tracking files...')
tracking = pd.concat(
    (pd.read_csv(f) for f in tracking_files),
    ignore_index=True
)
print(f'Tracking rows: {len(tracking):,}')

# ── 2. Identify pass plays & receivers ────────────────────────────────────
pass_plays = plays[
    plays['passResult'].isin(['C', 'I', 'IN'])
][['gameId', 'playId', 'passResult']].copy()

receivers = pff[
    pff['pff_role'] == 'Pass Route'
][['gameId', 'playId', 'nflId']].copy()
receivers = receivers.merge(pass_plays, on=['gameId', 'playId'])

rec_set = set(zip(receivers['gameId'], receivers['playId'], receivers['nflId']))

# ── 3. Helper functions ────────────────────────────────────────────────────
def abs_angular_diff(a, b):
    diff = (a - b + 180) % 360 - 180
    return abs(diff)

def compute_cut_angle(dir_series, window=5):
    dirs = dir_series.values
    max_cut = 0.0
    cut_frame_idx = 0
    for i in range(window, len(dirs)):
        change = abs_angular_diff(dirs[i], dirs[i - window])
        if change > max_cut:
            max_cut = change
            cut_frame_idx = i
    return max_cut, cut_frame_idx

# ── 4. Build route-level metrics ───────────────────────────────────────────
records = []

for (gameId, playId), play_track in tracking.groupby(['gameId', 'playId']):
    if gameId not in plays['gameId'].values:
        continue

    snap_rows   = play_track[play_track['event'] == 'ball_snap']
    arrive_rows = play_track[
        play_track['event'].isin([
            'pass_outcome_caught', 'pass_outcome_incomplete',
            'pass_outcome_interception', 'pass_arrived'
        ])
    ]

    if snap_rows.empty or arrive_rows.empty:
        continue

    snap_frame   = int(snap_rows['frameId'].iloc[0])
    arrive_frame = int(arrive_rows['frameId'].iloc[0])

    if arrive_frame <= snap_frame:
        continue

    for nflId in play_track['nflId'].dropna().unique():
        if (gameId, playId, nflId) not in rec_set:
            continue

        rec_track = play_track[
            (play_track['nflId'] == nflId) &
            (play_track['frameId'] >= snap_frame) &
            (play_track['frameId'] <= arrive_frame)
        ].sort_values('frameId').copy()

        if len(rec_track) < 6:
            continue
        rec_track = rec_track.dropna(subset=['dir', 'x', 'y', 's'])
        if len(rec_track) < 6:
            continue

        cut_angle, _ = compute_cut_angle(rec_track['dir'], window=5)

        records.append({
            'gameId':     gameId,
            'playId':     playId,
            'nflId':      nflId,
            'cut_angle':  cut_angle,
        })

metrics = pd.DataFrame(records)
metrics = metrics.merge(
    receivers[['gameId', 'playId', 'nflId', 'passResult']],
    on=['gameId', 'playId', 'nflId']
)
metrics = metrics.merge(
    players[['nflId', 'displayName', 'officialPosition']],
    on='nflId', how='left'
)

print(f'Route-level metrics: {len(metrics):,} routes')

# ── 5. Cut-angle buckets ───────────────────────────────────────────────────
bins   = [0, 30, 60, 90, 120, 150, 180]
labels = ['0–30°', '31–60°', '61–90°', '91–120°', '121–150°', '151–180°']
metrics['cut_bucket'] = pd.cut(
    metrics['cut_angle'], bins=bins, labels=labels, right=True
)
metrics['completed'] = (metrics['passResult'] == 'C').astype(int)

# ── 6. Select top receivers by route volume ────────────────────────────────
# Minimum 20 routes; keep top 15 by route count
MIN_ROUTES = 20
TOP_N      = 15

rec_counts = (
    metrics.groupby('nflId')
    .agg(routes=('playId', 'count'), name=('displayName', 'first'))
    .query('routes >= @MIN_ROUTES')
    .sort_values('routes', ascending=False)
    .head(TOP_N)
)

top_metrics = metrics[metrics['nflId'].isin(rec_counts.index)].copy()

# ── 7. Per-receiver: modal cut bucket + completion rate at that bucket ─────
def receiver_summary(grp):
    bucket_counts = grp['cut_bucket'].value_counts()
    modal_bucket  = bucket_counts.idxmax()
    at_bucket     = grp[grp['cut_bucket'] == modal_bucket]
    comp_rate     = at_bucket['completed'].mean()
    n_routes      = len(at_bucket)
    total_routes  = len(grp)
    return pd.Series({
        'modal_bucket': modal_bucket,
        'comp_rate':    comp_rate,
        'n_at_bucket':  n_routes,
        'total_routes': total_routes,
    })

summary = (
    top_metrics.groupby(['nflId', 'displayName'])
    .apply(receiver_summary)
    .reset_index()
    .sort_values('comp_rate', ascending=False)
)

print(summary[['displayName', 'modal_bucket', 'comp_rate', 'n_at_bucket']].to_string(index=False))

# ── 8. Plot ────────────────────────────────────────────────────────────────
# Color bars by cut-angle bucket
bucket_colors = {
    '0–30°':    '#3a86ff',
    '31–60°':   '#8338ec',
    '61–90°':   '#ff006e',
    '91–120°':  '#fb5607',
    '121–150°': '#ffbe0b',
    '151–180°': '#06d6a0',
}

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

x     = np.arange(len(summary))
bar_w = 0.65

bars = ax.bar(
    x,
    summary['comp_rate'] * 100,
    width=bar_w,
    color=[bucket_colors.get(str(b), ACCENT_COLOR) for b in summary['modal_bucket']],
    edgecolor='#0f3460',
    linewidth=0.8,
    zorder=3,
)

# ── Annotate each bar ──────────────────────────────────────────────────────
for i, (bar, row) in enumerate(zip(bars, summary.itertuples())):
    # Completion % inside bar
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 2.5,
        f'{row.comp_rate * 100:.1f}%',
        ha='center', va='top',
        fontsize=8.5, fontweight='bold', color='white', zorder=5,
    )
    # Sample size below the x-axis label
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        -4.5,
        f'n={row.n_at_bucket}',
        ha='center', va='top',
        fontsize=7.5, color='#aaaaaa', zorder=5,
    )

# ── Axes formatting ────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(
    summary['displayName'],
    rotation=30, ha='right', fontsize=9, color='#e0e0e0'
)
ax.set_ylabel('Completion Rate (%)', fontsize=11)
ax.set_ylim(0, 105)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#0f3460')
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor('#0f3460')

# ── Legend for cut-angle buckets ───────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=c, label=lbl)
    for lbl, c in bucket_colors.items()
]
ax.legend(
    handles=legend_patches,
    title='Modal Cut-Angle Bucket',
    title_fontsize=9,
    fontsize=8,
    loc='upper right',
    framealpha=0.3,
    edgecolor='#0f3460',
    facecolor='#1a1a2e',
    labelcolor='#e0e0e0',
)

ax.set_title(
    'Top Receivers: Most Common Cut-Angle Bucket & Completion Rate at That Cut\n'
    '(bar color = modal cut-angle range; height = completion rate; n = routes in that bucket)',
    fontsize=11, color='#e0e0e0', pad=14,
)

plt.tight_layout()
out_path = 'top_receivers_cut_vs_completion.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.close()
