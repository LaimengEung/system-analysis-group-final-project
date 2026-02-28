"""
Trajectory Distribution Grid — 16×16 or 32×32 Heatmap
=======================================================
Reads route data from route_cache.json (produced by visualize_trajectories.py)
and the original CSV to build a geographic grid.

Each cell in the grid is coloured by how many trajectory points pass through it.
White = 0 trips  →  Deep colour = maximum frequency.

Outputs: trajectory_grid.html  (self-contained, embeds all charts as base64 PNG)

Run:
    python trajectory_grid.py

Change GRID_SIZE below to switch between 16 and 32.

Requirements:
    pip install pandas matplotlib seaborn
"""

import os
import json
import base64
import io

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                           # headless — no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ─── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE    = "synthetic_data.csv"
CACHE_FILE  = "route_cache.json"
OUTPUT_FILE = "trajectory_grid.html"

GRID_SIZE   = 16          # change to 16 for a coarser grid

# Colour per vehicle (same palette as the map)
VEHICLE_COLORS = {
    "EV Car":    "#1E90FF",   # blue
    "Motor Dup": "#FF8C00",   # orange
    "Rickshaw":  "#2ECC71",   # green
    "Remork":    "#E74C3C",   # red
}

# ORS profile → vehicle types that use it (for cache key matching)
PROFILE_MAP = {
    "EV Car":    "driving-car",
    "Motor Dup": "cycling-electric",
    "Rickshaw":  "cycling-regular",
    "Remork":    "cycling-regular",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def make_colormap(hex_color: str):
    """Create a white → hex_color linear colormap."""
    return mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#ffffff", hex_color]
    )


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ─────────────────────────────────────────────────────────────────────────────
# Grid building
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(points: list[tuple], lat_min, lat_max, lon_min, lon_max,
               n: int) -> np.ndarray:
    """
    Given a list of (lat, lon) points and a bounding box, return an n×n
    numpy array where each cell contains the number of points that fall
    inside it.

    Row 0 = northernmost band (top of map), col 0 = westernmost band (left).
    """
    grid = np.zeros((n, n), dtype=float)

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if lat_range == 0 or lon_range == 0:
        return grid

    for lat, lon in points:
        col = int((lon - lon_min) / lon_range * n)
        row = int((lat_max - lat) / lat_range * n)   # flip: north = row 0
        # clamp to valid indices
        col = min(max(col, 0), n - 1)
        row = min(max(row, 0), n - 1)
        grid[row, col] += 1

    return grid


def collect_points_from_cache(cache: dict, vtypes: list[str]) -> list[tuple]:
    """
    Pull all [lat, lon] coordinate points from cached routes that belong
    to the requested vehicle types.

    Cache keys contain the ORS profile as the last segment separated by '|'.
    """
    profiles = {PROFILE_MAP[v] for v in vtypes if v in PROFILE_MAP}
    points = []

    for key, route in cache.items():
        if not route:
            continue
        key_profile = key.split("|")[-1]
        if key_profile in profiles:
            for pt in route:
                points.append((pt[0], pt[1]))   # lat, lon

    return points


def collect_od_points(df: pd.DataFrame, vtypes: list[str]) -> list[tuple]:
    """Fallback: collect origin + destination points from the CSV."""
    sub = df[df["vehicle_type"].isin(vtypes)]
    points = []
    for _, row in sub.iterrows():
        points.append((row["dept_lat"], row["dept_lon"]))
        points.append((row["arr_lat"],  row["arr_lon"]))
    return points


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(grid: np.ndarray, title: str, hex_color: str,
                 n: int, show_values: bool = False) -> str:
    """
    Draw a single heatmap and return it as a base64 PNG string.
    show_values=True prints the count in each cell (only practical for 16×16).
    """
    cmap = make_colormap(hex_color)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("#f8f8f8")

    sns.heatmap(
        grid,
        ax=ax,
        cmap=cmap,
        linewidths=0.3,
        linecolor="#dddddd",
        cbar=True,
        annot=show_values and (n <= 16),   # only annotate on 16×16
        fmt=".0f" if show_values else "",
        annot_kws={"size": 6},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("← West  ·  Longitude  ·  East →", fontsize=8, labelpad=6)
    ax.set_ylabel("← South  ·  Latitude  ·  North →", fontsize=8, labelpad=6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Colour-bar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Trip point density", fontsize=8)

    return fig_to_base64(fig)


def plot_combined_heatmap(grids: dict, all_points: list[tuple],
                          lat_min, lat_max, lon_min, lon_max,
                          n: int) -> str:
    """
    Draw the combined (all vehicles) heatmap using a red-purple gradient.
    """
    combined = build_grid(all_points, lat_min, lat_max, lon_min, lon_max, n)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "combined", ["#ffffff", "#c0392b", "#6c1a4a"]
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#f8f8f8")

    sns.heatmap(
        combined,
        ax=ax,
        cmap=cmap,
        linewidths=0.25,
        linecolor="#eeeeee",
        cbar=True,
        annot=(n <= 16),
        fmt=".0f",
        annot_kws={"size": 5.5},
    )

    ax.set_title(f"All Vehicles — {n}×{n} Trajectory Distribution",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("← West  ·  Longitude  ·  East →", fontsize=9, labelpad=6)
    ax.set_ylabel("← South  ·  Latitude  ·  North →", fontsize=9, labelpad=6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Total trip point density", fontsize=9)

    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def build_html(combined_b64: str, per_vehicle: list[tuple]) -> str:
    """
    Assemble a self-contained HTML page embedding all charts.
    per_vehicle: list of (vehicle_name, hex_color, base64_png)
    """

    def img_tag(b64: str, alt: str) -> str:
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="{alt}" style="max-width:100%;border-radius:6px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.12);">'
        )

    per_vehicle_html = ""
    for vname, color, b64 in per_vehicle:
        per_vehicle_html += f"""
        <div class="card">
            <h3 style="color:{color};">{vname}</h3>
            {img_tag(b64, vname)}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trajectory Distribution Grid — {GRID_SIZE}×{GRID_SIZE}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f0f2f5;
      color: #222;
      padding: 28px 20px 48px;
    }}
    h1 {{
      text-align: center;
      font-size: 1.6rem;
      margin-bottom: 6px;
      color: #1a1a2e;
    }}
    .subtitle {{
      text-align: center;
      font-size: 0.9rem;
      color: #555;
      margin-bottom: 32px;
    }}
    .section-title {{
      font-size: 1.15rem;
      font-weight: 600;
      color: #333;
      margin: 36px 0 16px;
      padding-left: 4px;
      border-left: 4px solid #c0392b;
      padding-left: 10px;
    }}
    .combined-wrap {{
      display: flex;
      justify-content: center;
      margin-bottom: 12px;
    }}
    .combined-wrap img {{
      max-width: 680px;
      width: 100%;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      padding: 16px 16px 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    .card h3 {{
      font-size: 1rem;
      margin-bottom: 10px;
    }}
    footer {{
      text-align: center;
      margin-top: 48px;
      font-size: 0.78rem;
      color: #aaa;
    }}
  </style>
</head>
<body>
  <h1>📊 Trajectory Distribution — {GRID_SIZE}×{GRID_SIZE} Grid</h1>
  <p class="subtitle">
    Phnom Penh ride-hail data &nbsp;|&nbsp;
    Brighter cell = more trip points passing through that area
  </p>

  <p class="section-title">All Vehicles Combined</p>
  <div class="combined-wrap">
    {img_tag(combined_b64, "Combined heatmap")}
  </div>

  <p class="section-title">Per Vehicle Type</p>
  <div class="grid">
    {per_vehicle_html}
  </div>

  <footer>
    Generated by trajectory_grid.py &nbsp;·&nbsp;
    Grid size: {GRID_SIZE}×{GRID_SIZE} &nbsp;·&nbsp;
    Source: {CSV_FILE}
  </footer>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    n = GRID_SIZE
    print(f"Building {n}×{n} trajectory distribution grid...")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_FILE)
    cache = load_cache()
    print(f"  CSV rows   : {len(df)}")
    print(f"  Cache entries: {len(cache)}")

    # ── Compute geographic bounding box ───────────────────────────────────────
    lat_min = min(df["dept_lat"].min(), df["arr_lat"].min())
    lat_max = max(df["dept_lat"].max(), df["arr_lat"].max())
    lon_min = min(df["dept_lon"].min(), df["arr_lon"].min())
    lon_max = max(df["dept_lon"].max(), df["arr_lon"].max())

    # Add small padding so edge points don't get clipped
    pad_lat = (lat_max - lat_min) * 0.02
    pad_lon = (lon_max - lon_min) * 0.02
    lat_min -= pad_lat;  lat_max += pad_lat
    lon_min -= pad_lon;  lon_max += pad_lon

    print(f"  Bounding box: lat [{lat_min:.4f}, {lat_max:.4f}]  "
          f"lon [{lon_min:.4f}, {lon_max:.4f}]")

    # ── Collect points per vehicle ────────────────────────────────────────────
    all_points = []
    per_vehicle_charts = []

    for vtype, hex_color in VEHICLE_COLORS.items():
        # Prefer full route points from cache; fall back to O/D points
        pts = collect_points_from_cache(cache, [vtype])
        source = "route cache"

        if not pts:
            pts = collect_od_points(df, [vtype])
            source = "O/D points (no cache)"

        print(f"  {vtype:10s}: {len(pts):6d} points  ({source})")
        all_points.extend(pts)

        grid = build_grid(pts, lat_min, lat_max, lon_min, lon_max, n)
        b64  = plot_heatmap(
            grid,
            title=f"{vtype}  —  {n}×{n}",
            hex_color=hex_color,
            n=n,
            show_values=True,
        )
        per_vehicle_charts.append((vtype, hex_color, b64))

    # ── Combined chart ────────────────────────────────────────────────────────
    print(f"\n  Total points (all vehicles): {len(all_points)}")
    combined_b64 = plot_combined_heatmap(
        {}, all_points, lat_min, lat_max, lon_min, lon_max, n
    )

    # ── Write HTML ────────────────────────────────────────────────────────────
    html = build_html(combined_b64, per_vehicle_charts)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Saved → {OUTPUT_FILE}")
    print("   Open it in any web browser to view the grid.")


if __name__ == "__main__":
    main()
