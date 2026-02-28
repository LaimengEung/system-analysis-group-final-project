"""
Trajectory Visualization — OpenRouteService + Folium
=====================================================
Generates an interactive HTML map showing ride-hail trajectories
in Phnom Penh with frequency-based color intensity (width + opacity).

Vehicle layers (togglable):
  ⚡ EV Car    → driving-car        → Blue
  🏍 Motor Dup → cycling-electric   → Orange
  🛺 Rickshaw  → cycling-regular    → Green
  🛻 Remork    → cycling-regular    → Red

Run:
  python visualize_trajectories.py

Requirements:
  pip install folium pandas requests python-dotenv
"""

import os
import json
import time
import pandas as pd
import requests
import folium
from collections import defaultdict
from dotenv import load_dotenv

# ─── Load API Key ──────────────────────────────────────────────────────────────
load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")

# ─── File Paths ────────────────────────────────────────────────────────────────
CSV_FILE    = "synthetic_ride_hail_phnom_penh.csv"
CACHE_FILE  = "route_cache.json"
OUTPUT_FILE = "trajectory_map.html"

# ─── ORS Rate Limit Settings ───────────────────────────────────────────────────
# Free tier: 40 requests/min, ~500 requests/day
# Batch size of 35 with 65s delay keeps us safely under 40 req/min.
BATCH_SIZE  = 35
BATCH_DELAY = 65    # seconds between batches

# ─── Segment Deduplication ─────────────────────────────────────────────────────
# Coordinates rounded to this many decimal places to detect shared road segments
# (4 decimal places ≈ ~11 metres precision — good for road-level matching)
COORD_ROUND = 4

# ─── Visual Style Ranges ───────────────────────────────────────────────────────
MIN_WEIGHT  = 2     # line width at lowest frequency
MAX_WEIGHT  = 9     # line width at highest frequency
MIN_OPACITY = 0.12  # transparency at lowest frequency
MAX_OPACITY = 1.0   # full brightness at highest frequency

# ─── Vehicle Configuration ─────────────────────────────────────────────────────
# Keys must match vehicle_type values in the CSV exactly.
VEHICLE_CONFIG = {
    "EV Car":    {"profile": "driving-car",      "color": "#1E90FF", "icon": "⚡"},
    "Motor Dup": {"profile": "cycling-electric", "color": "#FF8C00", "icon": "🏍"},
    "Rickshaw":  {"profile": "cycling-regular",  "color": "#2ECC71", "icon": "🛺"},
    "Remork":    {"profile": "cycling-regular",  "color": "#E74C3C", "icon": "🛻"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    """Load existing route cache from disk, or return empty dict."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    """Persist route cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def make_cache_key(origin: tuple, dest: tuple, profile: str) -> str:
    """
    Unique key for a (origin, destination, profile) combination.
    Rickshaw and Remork both use cycling-regular, so they automatically
    share cached routes when origin/dest match.
    """
    return (
        f"{round(origin[0], 5)},{round(origin[1], 5)}"
        f"|{round(dest[0], 5)},{round(dest[1], 5)}"
        f"|{profile}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ORS Route Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_route(origin: tuple, dest: tuple, profile: str, cache: dict):
    """
    Fetch a route polyline from ORS for the given origin, destination, profile.

    Returns a list of [lat, lon] coordinate pairs, or None on failure.
    Results are cached in `cache` (passed by reference).

    ORS expects [longitude, latitude] order; we convert to [lat, lon] on return.
    """
    key = make_cache_key(origin, dest, profile)

    if key in cache:
        return cache[key]   # cache hit — no API call needed

    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type":  "application/json",
    }
    body = {
        "coordinates": [
            [origin[1], origin[0]],   # ORS uses [lon, lat]
            [dest[1],   dest[0]],
        ]
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=15)

        if resp.status_code == 200:
            raw_coords = resp.json()["features"][0]["geometry"]["coordinates"]
            latlon = [[c[1], c[0]] for c in raw_coords]   # → [lat, lon]
            cache[key] = latlon
            return latlon

        elif resp.status_code == 429:
            # Rate limited — wait and retry once
            print("  ⚠  Rate limited (429). Waiting 70s before retrying...")
            time.sleep(70)
            return fetch_route(origin, dest, profile, cache)   # single retry

        else:
            print(f"  ⚠  ORS error {resp.status_code}: {resp.text[:120]}")
            return None

    except requests.exceptions.Timeout:
        print(f"  ⚠  Request timed out for {key[:50]}")
        return None
    except Exception as e:
        print(f"  ⚠  Unexpected error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Segment Frequency Counting
# ─────────────────────────────────────────────────────────────────────────────

def round_point(pt) -> tuple:
    """Round a [lat, lon] point to COORD_ROUND decimals for segment matching."""
    return (round(pt[0], COORD_ROUND), round(pt[1], COORD_ROUND))


def count_segments(routes: list) -> dict:
    """
    Given a list of route polylines, count how many routes pass through
    each road segment.

    A segment is a pair of consecutive rounded coordinate points.
    Direction is normalised (sorted) so A→B == B→A.
    Each route contributes at most 1 count per unique segment
    (prevents a single long route from inflating a segment's count).

    Returns: {segment_tuple → int frequency}
    """
    seg_count = defaultdict(int)

    for route in routes:
        if not route:
            continue
        seen_in_this_route = set()
        for i in range(len(route) - 1):
            seg = tuple(sorted([round_point(route[i]), round_point(route[i + 1])]))
            if seg not in seen_in_this_route:
                seen_in_this_route.add(seg)
                seg_count[seg] += 1

    return dict(seg_count)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation Helper
# ─────────────────────────────────────────────────────────────────────────────

def normalize(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    """Linearly map `value` from [lo, hi] into [out_lo, out_hi]."""
    if hi == lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Validate API key ──────────────────────────────────────────────────────
    if not ORS_API_KEY or ORS_API_KEY == "your_api_key_here":
        raise ValueError(
            "ORS_API_KEY is not set!\n"
            "Open the .env file and replace 'your_api_key_here' with your key.\n"
            "Get one free at: https://openrouteservice.org/dev/#/signup"
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    print(f"  → {len(df)} trips loaded")

    cache = load_cache()
    cache_hits_start = len(cache)

    # ── Collect trips per vehicle ─────────────────────────────────────────────
    all_trips = []   # list of (vehicle_type, origin, dest, profile)

    for _, row in df.iterrows():
        vtype = row["vehicle_type"]
        if vtype not in VEHICLE_CONFIG:
            continue
        all_trips.append((
            vtype,
            (row["dept_lat"], row["dept_lon"]),
            (row["arr_lat"],  row["arr_lon"]),
            VEHICLE_CONFIG[vtype]["profile"],
        ))

    total       = len(all_trips)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    est_minutes = round((num_batches * BATCH_DELAY) / 60, 1)

    print(f"\nFetching routes for {total} trips")
    print(f"  Batch size : {BATCH_SIZE}  |  Delay between batches: {BATCH_DELAY}s")
    print(f"  Estimated time (worst case, no cache): ~{est_minutes} min\n")

    # ── Fetch routes in batches ───────────────────────────────────────────────
    vehicle_routes = defaultdict(list)   # {vehicle_type → [route_polyline, ...]}

    for batch_idx in range(0, total, BATCH_SIZE):
        batch = all_trips[batch_idx: batch_idx + BATCH_SIZE]

        for vtype, origin, dest, profile in batch:
            coords = fetch_route(origin, dest, profile, cache)
            vehicle_routes[vtype].append(coords)

        save_cache(cache)   # persist after every batch

        done = min(batch_idx + BATCH_SIZE, total)
        new_hits = len(cache) - cache_hits_start
        print(f"  [{done:>4}/{total}]  cache size: {len(cache)}  (+{new_hits} new routes fetched)")

        if done < total:
            print(f"  Waiting {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)

    # ── Build Folium map ──────────────────────────────────────────────────────
    print("\nBuilding map...")
    center = [df["dept_lat"].mean(), df["dept_lon"].mean()]
    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles="CartoDB positron",      # light/white background
    )

    for vtype, config in VEHICLE_CONFIG.items():
        routes = vehicle_routes.get(vtype, [])
        valid  = [r for r in routes if r]

        if not valid:
            print(f"  ⚠  {vtype}: no valid routes, skipping layer")
            continue

        color     = config["color"]
        icon      = config["icon"]
        seg_count = count_segments(valid)

        if not seg_count:
            continue

        min_c = min(seg_count.values())
        max_c = max(seg_count.values())

        # Pre-compute style for every unique segment
        seg_style = {
            seg: (
                normalize(cnt, min_c, max_c, MIN_OPACITY, MAX_OPACITY),  # opacity
                normalize(cnt, min_c, max_c, MIN_WEIGHT,  MAX_WEIGHT),   # width
                cnt,                                                       # raw count
            )
            for seg, cnt in seg_count.items()
        }

        layer = folium.FeatureGroup(name=f"{icon} {vtype}", show=True)

        drawn = set()   # draw each unique segment only once
        for route in valid:
            for i in range(len(route) - 1):
                seg = tuple(sorted([round_point(route[i]), round_point(route[i + 1])]))
                if seg in drawn:
                    continue
                drawn.add(seg)

                opacity, weight, cnt = seg_style[seg]

                folium.PolyLine(
                    locations=[list(seg[0]), list(seg[1])],
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    tooltip=f"{vtype} — segment used by {cnt} trip(s)",
                ).add_to(layer)

        layer.add_to(m)
        print(f"  ✓ {icon} {vtype:10s}  routes: {len(valid):4d}  unique segments: {len(drawn):5d}  "
              f"freq range: {min_c}–{max_c}")

    # ── Layer control + legend ────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    # Inline legend (bottom-right)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 10px;
        background-color: rgba(0,0,0,0.75);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        z-index: 9999;
        line-height: 1.8em;
    ">
        <b>Trajectory Frequency</b><br>
        Thicker &amp; more opaque = more trips<br><br>
        <span style="color:#1E90FF">&#9644;</span> ⚡ EV Car<br>
        <span style="color:#FF8C00">&#9644;</span> 🏍 Motor Dup<br>
        <span style="color:#2ECC71">&#9644;</span> 🛺 Rickshaw<br>
        <span style="color:#E74C3C">&#9644;</span> 🛻 Remork
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Save ──────────────────────────────────────────────────────────────────
    m.save(OUTPUT_FILE)
    print(f"\n✅ Map saved → {OUTPUT_FILE}")
    print("   Open it in any web browser to view.")


if __name__ == "__main__":
    main()
