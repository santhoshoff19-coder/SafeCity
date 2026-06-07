from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import json
import os
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="SafeCity Chicago API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────
LAT_MIN, LON_MIN = 41.64, -87.94
GRID_SIZE = 0.005
DATA_DIR  = "../Data"

# ── Load everything on startup ─────────────────────────
print("Loading data...")

risk_df  = pd.read_csv(f"{DATA_DIR}/risk_scores.csv")
neigh_df = pd.read_csv(f"{DATA_DIR}/neighbourhood_data.csv")

with open(f"{DATA_DIR}/lgb_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

print("Loading road network...")
G = ox.load_graphml(f"{DATA_DIR}/chicago_walk_network.graphml")

print("Loading edge dangers...")
edge_dangers = {}
for hour in range(24):
    path = f"{DATA_DIR}/edge_dangers/hour_{hour:02d}.pkl"
    with open(path, "rb") as f:
        edge_dangers[hour] = pickle.load(f)

print("Loading hourly maps...")
hourly_combined = {}
for hour in range(24):
    path = f"{DATA_DIR}/hourly_maps/hour_{hour:02d}.json"
    if os.path.exists(path):
        with open(path) as f:
            hourly_combined[hour] = json.load(f)

hourly_category = {}
for category in ['violent','sexual','weapons','property','drugs']:
    hourly_category[category] = {}
    for hour in range(24):
        path = f"{DATA_DIR}/hourly_maps/{category}/hour_{hour:02d}.json"
        if os.path.exists(path):
            with open(path) as f:
                hourly_category[category][hour] = json.load(f)

geolocator = Nominatim(user_agent="safecity_chicago_v2")

print("All loaded. API ready.")

# ── Helper functions ───────────────────────────────────
def get_safety_score(lat, lon, hour):
    lat_grid = int((lat - LAT_MIN) / GRID_SIZE)
    lon_grid = int((lon - LON_MIN) / GRID_SIZE)

    match = risk_df[
        (risk_df['lat_grid'] == lat_grid) &
        (risk_df['lon_grid'] == lon_grid) &
        (risk_df['hour']     == hour)
    ]

    if len(match) == 0:
        return 9.5, 'none', 0.0

    row = match.iloc[0]
    return (
        float(row['safety_score']),
        str(row['dominant_category']),
        float(row['avg_cluster_prob'])
    )


def build_weighted_graph_from_cache(hour, user_weights):
    dangers = edge_dangers[hour]
    G_w = G.copy()
    total = sum(user_weights.values()) or 1

    for u, v, k, data in G_w.edges(data=True, keys=True):
        length = data.get('length', 10)
        danger = dangers.get((u, v, k), 0)
        crime_weight = length * (1 + danger * 5)
        G_w[u][v][k]['crime_weight'] = crime_weight
        G_w[u][v][k]['danger']       = danger

    return G_w


def geocode(address):
    try:
        print(f"Geocoding: {address}")
        loc = geolocator.geocode(address)
        if loc:
            print(f"Found: {loc.latitude}, {loc.longitude}")
            return loc.latitude, loc.longitude
        print("Geocoding returned None")
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None


# ── Endpoints ──────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "SafeCity API running"}


@app.get("/heatmap")
def get_heatmap(hour: int = 22, category: str = "combined"):
    if category == "combined":
        data = hourly_combined.get(hour, [])
    else:
        data = hourly_category.get(category, {}).get(hour, [])
    return {"hour": hour, "category": category, "data": data}


@app.get("/route")
def get_route(
    start:    str,
    end:      str,
    hour:     int   = 22,
    violent:  float = 10,
    sexual:   float = 8,
    weapons:  float = 7,
    property: float = 5,
    drugs:    float = 3
):
    try:
        user_weights = {
            'violent':  violent,
            'sexual':   sexual,
            'weapons':  weapons,
            'property': property,
            'drugs':    drugs
        }

        print(f"\n=== ROUTE REQUEST ===")
        print(f"Start: {start}")
        print(f"End:   {end}")
        print(f"Hour:  {hour}")

        s_lat, s_lon = geocode(start)
        e_lat, e_lon = geocode(end)

        print(f"Start coords: {s_lat}, {s_lon}")
        print(f"End coords:   {e_lat}, {e_lon}")

        if not s_lat or not e_lat:
            return {"error": "Could not geocode one or both addresses"}

        print("Building weighted graph...")
        G_w = build_weighted_graph_from_cache(hour, user_weights)

        print("Finding nearest nodes...")
        s_node = ox.distance.nearest_nodes(G_w, s_lon, s_lat)
        e_node = ox.distance.nearest_nodes(G_w, e_lon, e_lat)
        print(f"Start node: {s_node}, End node: {e_node}")

        print("Computing safest path...")
        safest_path = nx.shortest_path(G_w, s_node, e_node, weight='crime_weight')
        print("Computing shortest path...")
        shortest_path = nx.shortest_path(G_w, s_node, e_node, weight='length')
        print(f"Safest path nodes: {len(safest_path)}, Shortest: {len(shortest_path)}")

        def path_to_coords(path):
            return [[G_w.nodes[n]['y'], G_w.nodes[n]['x']] for n in path]

        def path_stats(path):
            length, danger = 0, 0
            for u, v in zip(path[:-1], path[1:]):
                d = G_w[u][v]
                k = list(d.keys())[0]
                length += d[k].get('length', 10)
                danger += d[k].get('danger', 0)
            avg_danger = danger / max(len(path)-1, 1)
            return {
                'length_km':  round(length/1000, 2),
                'avg_safety': round((1-avg_danger)*10, 2)
            }

        result = {
            "start":   {"lat": s_lat, "lon": s_lon},
            "end":     {"lat": e_lat, "lon": e_lon},
            "safest":  {"coords": path_to_coords(safest_path),  "stats": path_stats(safest_path)},
            "shortest":{"coords": path_to_coords(shortest_path),"stats": path_stats(shortest_path)}
        }

        print("Route computed successfully.")
        return result

    except Exception as e:
        print(f"ERROR in route: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/neighbourhood")
def get_neighbourhood(
    min_rent:        float = 800,
    max_rent:        float = 2000,
    safety_w:        float = 10,
    affordability_w: float = 5,
    nightlife_w:     float = 3,
    family_w:        float = 3,
    walkability_w:   float = 3
):
    df = neigh_df.copy()

    df = df[
        (df['estimated_rent'] >= min_rent) &
        (df['estimated_rent'] <= max_rent)
    ]

    if len(df) == 0:
        return {"error": "No neighbourhoods in budget range"}

    def norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return series * 0
        return (series - mn) / (mx - mn)

    df['safety_norm']      = norm(df['avg_night_safety'])
    df['afford_norm']      = norm(df['estimated_rent'].max() - df['estimated_rent'])
    df['nightlife_norm']   = norm(df['nightlife'])
    df['family_norm']      = norm(df['family_friendly'])
    df['walkability_norm'] = norm(df['walkability'])

    total_w = (safety_w + affordability_w + nightlife_w + family_w + walkability_w) or 1

    df['score'] = (
        safety_w        * df['safety_norm'] +
        affordability_w * df['afford_norm'] +
        nightlife_w     * df['nightlife_norm'] +
        family_w        * df['family_norm'] +
        walkability_w   * df['walkability_norm']
    ) / total_w * 10

    df['score'] = df['score'].round(2)

    top5 = df.nlargest(5, 'score')[[
        'name', 'estimated_rent', 'avg_night_safety',
        'avg_day_safety', 'walkability', 'nightlife',
        'family_friendly', 'score'
    ]].to_dict(orient='records')

    return {"results": top5}


@app.get("/safety")
def get_point_safety(lat: float, lon: float, hour: int = 22):
    score, category, prob = get_safety_score(lat, lon, hour)
    return {
        "lat": lat, "lon": lon, "hour": hour,
        "safety_score": score,
        "dominant_category": category,
        "cluster_confidence": round(prob, 3)
    }