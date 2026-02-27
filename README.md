# 🛺 Ride-Hail Analysis — Phnom Penh

A data analysis project on a synthetic ride-hailing dataset based in Phnom Penh, Cambodia. This project covers descriptive, diagnostic, and visual analysis of ride patterns, pricing, vehicle types, and more.

---

## 📁 Project Structure

```
├── analysis.ipynb                    # Main analysis notebook
├── dataset.ipynb                     # Dataset generation & exploration
├── prediction.ipynb                  # Predictive modeling
├── folium.ipynb                      # Map visualization (Folium)
├── data_validator.ipynb              # Data validation
├── synthetic_ride_hail_phnom_penh.csv  # Synthetic dataset
├── trajectory_grid.py                # Trajectory grid generator
├── visualize_trajectories.py         # Trajectory visualization
│
├── polar_ride_by_hour.html           # Polar line chart (rides by hour)
├── polar_filled_ride_by_hour.html    # Polar bar chart (rides by hour)
├── time_series.html                  # Line chart (rides by hour & vehicle)
├── normalized_time_series.html       # Normalized line chart
├── combined_chart.html               # Combined chart view
├── ride_hail_routes_folium.html      # Folium route map
├── trajectory_map.html               # Trajectory map
├── trajectory_grid.html              # Trajectory grid (16×16)
└── route_cache.json                  # Cached route data
```

---

## 📊 Dataset

**File:** `synthetic_ride_hail_phnom_penh.csv`  
**Size:** ~2,168 trips  
**Period:** February 2026

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `trip_id` | int | Unique trip identifier |
| `date` | string | Trip date |
| `dept_lat/lon` | float | Departure coordinates |
| `arr_lat/lon` | float | Arrival coordinates |
| `vehicle_type` | string | EV Car, Motor Dup, Remork, Rickshaw |
| `request_time` | string | Time of ride request |
| `wait_time_min` | int | Wait time in minutes |
| `trip_distance_km` | float | Trip distance |
| `est_time_min` | int | Estimated trip time |
| `actual_time_min` | int | Actual trip time |
| `fare_usd` | float | Fare in USD |
| `tip_usd` | float | Tip in USD |
| `rating` | int | Passenger rating (1–5) |
| `surge_pricing` | string | Surge level (Low / Medium / High / Very High) |
| `weather` | string | Weather condition |

---

## 🔍 Analysis Overview

### 1. Descriptive Analysis
- Statistical summary of quantitative variables (`fare_usd`, `trip_distance_km`, `wait_time_min`, etc.)
- Distribution plots (histograms & count plots)
- Time-of-day classification:

| Label | Time Range |
|-------|-----------|
| Morning | 06:00 – 08:59 |
| Late Morning | 09:00 – 11:59 |
| Afternoon | 12:00 – 14:59 |
| Late Afternoon | 15:00 – 17:59 |
| Evening | 18:00 – 20:59 |
| Night | 21:00 – 23:59 |

---

### 2. Diagnostic Analysis

#### Quantitative vs Quantitative
- Trip distance vs Tip (correlation)
- Estimated time vs Actual time (by vehicle type)
- Trip distance vs Fare

#### Quantitative vs Qualitative
- Weather vs Actual ride time (boxplot)
- Weekday vs Weekend ride distribution (pie + bar)
- Surge pricing vs Fare & Wait time (boxplot + ANOVA)
- Vehicle type vs Tip amount (boxplot + ANOVA)
- Time of day vs Ride requests (polar & line charts)

#### Statistical Tests (ANOVA + Eta-Squared η²)
Effect size guide:

| Threshold | Effect |
|-----------|--------|
| η² ≥ 0.14 | Large |
| η² ≥ 0.06 | Medium |
| η² ≥ 0.01 | Small |
| η² < 0.01 | Negligible |

---

## 🗺️ Visualizations

| File | Description |
|------|-------------|
| `polar_ride_by_hour.html` | Polar line chart — ride volume by hour & vehicle type |
| `polar_filled_ride_by_hour.html` | Polar bar chart — ride volume by hour |
| `time_series.html` | Line series — rides by hour & vehicle type |
| `normalized_time_series.html` | Normalized ride patterns (0–1 scale) |
| `trajectory_map.html` | Trajectory map by vehicle type |
| `trajectory_grid.html` | 16×16 spatial trajectory grid |

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly pingouin folium
```

---

## 🚀 Getting Started

1. Clone or download this repository
2. Install dependencies (see above)
3. Open `analysis.ipynb` in Jupyter or VS Code
4. Run all cells from top to bottom

---

## ⚠️ Notes

- This dataset is **fully synthetic** — generated for academic purposes
- All statistical tests should be interpreted in the context of synthetic data
- No real personal or location data is used