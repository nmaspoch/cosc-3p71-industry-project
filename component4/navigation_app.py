import streamlit as st
import pickle
import networkx as nx
import folium
import osmnx as ox  
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import random
import time  # Added for execution time benchmarking

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from component2.graph import calculate_haversine_distance

def get_coords(G, node_id):
    """Returns (lon, lat) for a given node, compatible with both OSMnx and legacy formats."""
    node_data = G.nodes[node_id]
    if 'pos' in node_data:
        return node_data['pos']  
    return (float(node_data.get('x', 0)), float(node_data.get('y', 0)))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Smart City Navigation System",
    page_icon="🚗",
    layout="wide"
)

# ============================================================================
# CONSTANTS
# ============================================================================

SPEED_CONFIG = {
    'safe': 40,
    'minor_issues': 25,
    'major_issues': 15,
    'major_problems': 15
}
DEFAULT_SPEED = 30 

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_resource
def load_graph():
    """Load the graph and normalize attributes for the navigation app."""
    graph_path = Path('osm_road_network.graphml')
    
    if not graph_path.exists():
        st.error("❌ 'osm_road_network.graphml' not found. Please run graph.py first.")
        return None

    try:
        G_raw = ox.load_graphml(filepath=str(graph_path))
        G = ox.convert.to_digraph(G_raw, weight='weight')
        
        for u, v, data in G.edges(data=True):
            if 'length' in data:
                data['raw_distance'] = float(data['length'])
            elif 'raw_distance' in data:
                data['raw_distance'] = float(data['raw_distance'])
            else:
                data['raw_distance'] = 100.0 
                
            if 'road_condition' in data:
                data['condition'] = data['road_condition']
            
            for attr in ['weight', 'safety_penalty', 'yolo_probability']:
                if attr in data:
                    try:
                        data[attr] = float(data[attr])
                    except (ValueError, TypeError):
                        data[attr] = 1.0

        for node, data in G.nodes(data=True):
            for attr in ['x', 'y', 'lat', 'lon']:
                if attr in data:
                    data[attr] = float(data[attr])

        return G
        
    except Exception as e:
        st.error(f"Error loading or processing graph: {e}")
        return None

@st.cache_data
def load_metadata():
    """Load the original metadata"""
    metadata_path = Path(__file__).resolve().parent.parent / 'final_metadata.csv'
    return pd.read_csv(metadata_path)

# ============================================================================
# NAVIGATION FUNCTIONS
# ============================================================================

def get_node_coords(G, node):
    """Safely get lat/lon regardless of attribute names."""
    data = G.nodes[node]
    lat = data.get('y') or data.get('lat')
    lon = data.get('x') or data.get('lon')
    return float(lon), float(lat)

def find_nearest_node(G, target_lat, target_lon):
    """Find the closest node in graph to given coordinates"""
    min_dist = float('inf')
    nearest_node = None
    
    for node, data in G.nodes(data=True):
        lon, lat = get_node_coords(G, node)
        dist = calculate_haversine_distance(lat, lon, target_lat, target_lon)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node, min_dist

def calculate_route_safety(G, path):
    """Calculate overall safety classification for a route"""
    if len(path) < 2:
        return 'Safe', 0.1, 0, 'green'
    
    total_distance = 0
    weighted_safety = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            edge_data = G[u][v]
            distance = edge_data['raw_distance']
            penalty = edge_data['safety_penalty']
            total_distance += distance
            weighted_safety += distance * penalty
    
    avg_penalty = weighted_safety / total_distance if total_distance > 0 else 1.0
    
    if avg_penalty <= 1.2:
        classification, probability, color = 'Safe', (avg_penalty - 1.0) / 2.0, 'green'
    elif avg_penalty <= 2.0:
        classification, probability, color = 'Possibly Hazardous', 0.3 + (avg_penalty - 1.2) * 0.5, 'orange'
    else:
        classification, probability, color = 'Hazardous', min(0.7 + (avg_penalty - 2.0) * 0.3, 1.0), 'red'
    
    return classification, probability, total_distance, color

def calculate_travel_time(G, path):
    """Calculate estimated travel time in minutes."""
    if len(path) < 2: return 0.0
    total_time_hours = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            edge_data = G[u][v]
            distance_km = edge_data['raw_distance'] / 1000
            speed = SPEED_CONFIG.get(edge_data.get('condition', 'safe'), DEFAULT_SPEED)
            total_time_hours += (distance_km / speed)
    return total_time_hours * 60

def find_route(G, start_node, end_node, algorithm='dijkstra'):
    """Find the shortest weighted path and measure execution time."""
    start_time = time.perf_counter()  # Start timer
    try:
        if algorithm.lower() == 'a*':
            def heuristic(u, v):
                u_lon, u_lat = get_coords(G, u)
                v_lon, v_lat = get_coords(G, v)
                return calculate_haversine_distance(u_lat, u_lon, v_lat, v_lon)
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')
            path_length = nx.astar_path_length(G, start_node, end_node, heuristic=heuristic, weight='weight')
        else:
            path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
            path_length = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000  # Convert to ms
        
        classification, prob, dist, color = calculate_route_safety(G, path)
        return {
            'path': path, 'weighted_cost': path_length, 'actual_distance': dist,
            'travel_time': calculate_travel_time(G, path), 'safety_classification': classification,
            'safety_probability': prob, 'color': color, 'exists': True,
            'execution_time': execution_time_ms
        }
    except nx.NetworkXNoPath:
        return {'exists': False, 'execution_time': 0}

def find_alternative_routes(G, start_node, end_node, k=3, algorithm='dijkstra'):
    routes = []
    G_temp = G.copy()
    for _ in range(k):
        route = find_route(G_temp, start_node, end_node, algorithm=algorithm)
        if not route['exists']: break
        routes.append(route)
        for j in range(len(route['path']) - 1):
            u, v = route['path'][j], route['path'][j+1]
            if G_temp.has_edge(u, v): G_temp[u][v]['weight'] *= 10
            if G_temp.has_edge(v, u): G_temp[v][u]['weight'] *= 10
    return routes

# ============================================================================
# REAL-TIME DETECTION FUNCTIONS
# ============================================================================

def simulate_random_hazard(G, excluded_edges=None):
    if excluded_edges is None: excluded_edges = set()
    available_edges = [(u, v) for u, v, _ in G.edges(data=True) if (u, v) not in excluded_edges]
    if not available_edges: return None
    u, v = random.choice(available_edges)
    new_condition = random.choice(['major_issues', 'major_problems'])
    return {
        'edge': (u, v), 'new_condition': new_condition, 'old_condition': G[u][v].get('condition', 'safe'),
        'new_penalty': 3.0, 'old_penalty': G[u][v].get('safety_penalty', 1.0), 'timestamp': datetime.now().strftime("%H:%M:%S")
    }

def apply_hazard_to_graph(G, hazard_info):
    G_updated = G.copy()
    u, v = hazard_info['edge']
    for edge in [(u, v), (v, u)]:
        if G_updated.has_edge(*edge):
            G_updated[edge[0]][edge[1]]['condition'] = hazard_info['new_condition']
            G_updated[edge[0]][edge[1]]['safety_penalty'] = hazard_info['new_penalty']
            G_updated[edge[0]][edge[1]]['weight'] = G_updated[edge[0]][edge[1]]['raw_distance'] * hazard_info['new_penalty']
    return G_updated

def check_route_affected(route, hazard_info):
    u, v = hazard_info['edge']
    path = route['path']
    for i in range(len(path) - 1):
        if (path[i] == u and path[i+1] == v) or (path[i] == v and path[i+1] == u):
            return True
    return False

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_base_folium_map(G, center_lat, center_lon):
    return folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')

def add_markers_to_map(m, start_coords, end_coords):
    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
    folium.Marker(end_coords, popup="End", icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("🚗 Smart City Road Safety Navigation System")
    st.markdown("### Component 4: Intelligent Route Planning")
    
    with st.spinner("Loading navigation system..."):
        G = load_graph()
        df = load_metadata()
    
    st.sidebar.header("Navigation Controls")
    input_method = st.sidebar.radio("Select Input Method:", ["Preset Locations", "Select Nodes", "Random Points"])
    
    if input_method == "Preset Locations":
        locations = {f"Point {i}": (df.iloc[i]['latitude'], df.iloc[i]['longitude']) for i in [0, 15, 30, 45]}
        start_loc = st.sidebar.selectbox("Start Location:", list(locations.keys()), index=0)
        end_loc = st.sidebar.selectbox("End Location:", list(locations.keys()), index=2)
        start_lat, start_lon = locations[start_loc]
        end_lat, end_lon = locations[end_loc]
    elif input_method == "Select Nodes":
        all_nodes = list(G.nodes())
        start_node_select = st.sidebar.selectbox("Start Node:", all_nodes, index=0)
        end_node_select = st.sidebar.selectbox("End Node:", all_nodes, index=min(30, len(all_nodes)-1))
        start_lon, start_lat = get_node_coords(G, start_node_select)
        end_lon, end_lat = get_node_coords(G, end_node_select)
    else:
        if st.sidebar.button("Generate Random Points"):
            st.session_state['rnd'] = random.sample(range(len(df)), 2)
        rnd = st.session_state.get('rnd', [0, len(df)-1])
        start_lat, start_lon = df.iloc[rnd[0]]['latitude'], df.iloc[rnd[0]]['longitude']
        end_lat, end_lon = df.iloc[rnd[1]]['latitude'], df.iloc[rnd[1]]['longitude']

    num_routes = st.sidebar.slider("Number of Alternatives", 1, 3, 3)
    algorithm = st.sidebar.radio("Algorithm:", ["Dijkstra", "A*"])

    if st.sidebar.button("🔍 Calculate Routes", type="primary"):
        st.session_state['should_calculate'] = True

    if st.session_state.get('should_calculate', False):
        start_node, _ = find_nearest_node(G, start_lat, start_lon)
        end_node, _ = find_nearest_node(G, end_lat, end_lon)
        
        if start_node == end_node:
            st.error("Points too close!")
        else:
            routes = find_alternative_routes(G, start_node, end_node, k=num_routes, algorithm=algorithm)
            if not routes:
                st.error("No routes found!")
            else:
                # --- ROUTE SELECTION & DETAILED METRICS ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("Route Selection")
                selected_route_name = st.sidebar.radio(
                    "Select route to display on map:",
                    options=[f"Route {i+1}" for i in range(len(routes))],
                    index=0
                )
                selected_idx = int(selected_route_name.split()[-1]) - 1
                active_route = routes[selected_idx]

                # Detailed Metrics Row
                st.markdown(f"### 📊 Detailed Stats: {selected_route_name}")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Total Distance", f"{active_route['actual_distance']:.0f}m")
                with m2:
                    st.metric("Travel Time", f"{active_route['travel_time']:.1f} min")
                with m3:
                    st.metric("Safety Probability", f"{active_route['safety_probability']:.3f}")
                with m4:
                    st.metric("Safety Rating", active_route['safety_classification'])
                with m5:
                    st.metric("Algo Execution", f"{active_route['execution_time']:.3f} ms", help=f"Time taken by {algorithm} to find path")

                # Map Rendering
                m = create_base_folium_map(G, (start_lat+end_lat)/2, (start_lon+end_lon)/2)
                
                # Draw the specific selected route with safety color segments
                path = active_route['path']
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    u_lon, u_lat = get_coords(G, u)
                    v_lon, v_lat = get_coords(G, v)
                    
                    edge_data = G[u][v]
                    penalty = edge_data.get('safety_penalty', 1.0)
                    
                    # Segment coloring logic
                    if penalty <= 1.2:
                        seg_color = 'green'
                    elif penalty <= 2.0:
                        seg_color = 'orange'
                    else:
                        seg_color = 'red'
                        
                    folium.PolyLine(
                        [[u_lat, u_lon], [v_lat, v_lon]], 
                        color=seg_color, 
                        weight=8, 
                        opacity=0.9,
                        tooltip=f"Condition: {edge_data.get('condition', 'safe')}"
                    ).add_to(m)

                lon_s, lat_s = get_coords(G, start_node)
                lon_e, lat_e = get_coords(G, end_node)
                add_markers_to_map(m, [lat_s, lon_s], [lat_e, lon_e])
                
                # Single Route Legend
                legend_html = f'''
                <div style="position: fixed; bottom: 50px; right: 50px; width: 180px; background: white; border:2px solid grey; z-index:9999; padding: 10px; font-size:12px;">
                <b>Viewing: {selected_route_name}</b><br>
                <i style="color:green">━</i> Safe<br><i style="color:orange">━</i> Caution<br><i style="color:red">━</i> Hazardous
                </div>'''
                m.get_root().html.add_child(folium.Element(legend_html))
                st_folium(m, width=1200, height=500)

                # Real-Time Hazard Detection
                st.markdown("---")
                st.subheader("🚨 Hazard Simulation")
                if st.button("🎲 Simulate Hazard Detection"):
                    h = simulate_random_hazard(G)
                    if h:
                        st.warning(f"Hazard detected: {h['new_condition']} between node {h['edge'][0]} and {h['edge'][1]}")
                        if check_route_affected(active_route, h):
                            # Calculate Delay Impact
                            G_affected = apply_hazard_to_graph(G, h)
                            current_time = active_route['travel_time']
                            
                            # Find new travel time on the same path with new condition
                            new_time = calculate_travel_time(G_affected, active_route['path'])
                            delay_impact = new_time - current_time
                            
                            st.error(f"🚨 YOUR CURRENT ROUTE IS AFFECTED!")
                            st.write(f"**Delay Impact:** +{delay_impact:.1f} minutes to your current trip.")
                            st.info("💡 Re-routing recommended to find a safer or faster alternative.")

if __name__ == "__main__":
    main()