import math
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points

# Constants for data paths
EXCLUDED_ZONES = {"Chamorro", "Samoa", "Atlantic"}
EXCLUDED_STATES = {
    "American Samoa", "Guam", "Commonwealth of the Northern Mariana Islands",
    "Puerto Rico", "United States Virgin Islands"
}

# Data Loading Functions

def load_parks(csv_path: str) -> gpd.GeoDataFrame:
    """
    Load park coordinates from CSV and return a GeoDataFrame with WGS84 CRS.
    """
    df = pd.read_csv(csv_path)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    parks = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs='EPSG:4326'
    )
    return parks


def load_states(states_path: str) -> gpd.GeoDataFrame:
    """
    Load US states from shapefile, filter exclusions, return GeoDataFrame.
    """
    states = gpd.read_file(states_path)
    states = states[~states['name'].isin(EXCLUDED_STATES)]
    return states


def load_timezones(timezones_path: str) -> gpd.GeoDataFrame:
    """
    Load time zones from shapefile, filter exclusions, return GeoDataFrame.
    """
    tz = gpd.read_file(timezones_path)
    tz = tz[~tz['zone'].isin(EXCLUDED_ZONES)]
    return tz

# calculation for topology

def calc_topology(state, timezone, direction):
    """
    Input: state row and timezone row, direction ('state' or 'zone', meaning finding relation
    from state to timezone or vice versa).
    Output: list of relations between state and timezone.
    """
    upper = 0.98
    lower = 0.02
    geom_state = state.geometry
    geom_other = timezone.geometry
    inter = geom_state.intersection(geom_other)
    ia = inter.area
    area_state = geom_state.area
    upper_area = upper * area_state
    lower_area = lower * area_state

    rels = []
    if ia == 0:
        if state['name'] == "Utah" and timezone.zone == "Pacific":
            rels.append("touch")
        elif state['name'] == "Montana" and timezone.zone == "Central":
            rels.append("touch")
        else:
            rels.append("disjoint")
    if 0 < ia <= lower_area:
        rels.append("touch")
    if lower_area < ia <= upper_area:
        rels.append("overlaps")
    if ia >= upper_area:
        special = {"Iowa", "Missouri", "Arkansas", "West Virginia"}
        if state['name'] in special:
            if direction == "state":
                rels.append("within")
            else:
                rels.append("contains")
        elif state['name'] == "Alaska":
            if direction == "zone":
                rels.append("within")
            else:
                rels.append("contains")
        else:
            if direction == "state":
                rels.append("covered by")
            else:
                rels.append("covers")

    return sorted(set(rels))


# calculateion for direction
def calc_direction(parks_gdf, row) -> list:
    """
    Given row with 'park1' and 'park2', compute compass direction neighbors.
    """
    # Lookup coordinates
    p1 = parks_gdf.loc[parks_gdf['name'] == row['park1']].iloc[0]
    p2 = parks_gdf.loc[parks_gdf['name'] == row['park2']].iloc[0]
    lat1, lon1 = p1.latitude, p1.longitude
    lat2, lon2 = p2.latitude, p2.longitude

    # Calculate bearing
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - \
        math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360

    # Map to compass directions including neighbors
    directions = ['North', 'Northeast', 'East', 'Southeast',
                  'South', 'Southwest', 'West', 'Northwest']
    idx = int(((bearing + 22.5) % 360) / 45)
    return [
        directions[(idx - 1) % len(directions)],
        directions[idx],
        directions[(idx + 1) % len(directions)]
    ]


# calculation for distance

def calc_park_to_park_distance(lat1, lon1, lat2, lon2) -> float:
    """
    Haversine distance between two lat/lon points (km).
    """
    R=6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))



def calc_park_to_state_distance(park_name, parks_gdf, state_name, states_gdf) -> float:
    """
    Compute nearest distance (km) from a park point to state boundary.
    """
    park = parks_gdf[parks_gdf['name']==park_name]
    if park.empty:
        raise ValueError(f"Park not found: {park_name}")
    state = states_gdf[states_gdf['name']==state_name]
    if state.empty:
        raise ValueError(f"State not found: {state_name}")
    p_geom = park.geometry.iloc[0]
    s_geom = state.geometry.iloc[0]
    if s_geom.contains(p_geom):
        return 0.0
    boundary = s_geom.boundary
    nearest = nearest_points(p_geom, boundary)[1]
    return calc_park_to_park_distance(p_geom.y, p_geom.x, nearest.y, nearest.x)



