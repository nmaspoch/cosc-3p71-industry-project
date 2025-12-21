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
import time

# Setup project paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from component2.graph import calculate_haversine_distance

# ============================================================================
# DATA LOADING (FIXED FOR TYPE ERROR)
# ============================================================================

@st.cache_resource
def load_graph():
    """Load graph and ensure weights are numeric to prevent TypeErrors."""
    graph_path = Path('osm_road_network.graphml')
    if not graph_path.exists():
        st.error("❌ 'osm_road_network.graphml' not found.")
        return None
    try:
        G_raw = ox.load_graphml(filepath=str(graph_path))
        G = ox.convert.to_digraph(G_raw, weight='weight')
        
        for u, v, data in G.edges(data=True):
            data['raw_distance'] = float(data.get('length', 100.0))
            data['condition'] = data.get('road_condition', 'safe')
            try:
                data['weight'] = float(data.get('weight', data['raw_distance']))
                data['safety_penalty'] = float(data.get('safety_penalty', 1.0))
            except (ValueError, TypeError):
                data['weight'] = float(data['raw_distance'])
                data['safety_penalty'] = 1.0
                
        for node, data in G.nodes(data=True):
            for attr in ['x', 'y', 'lat', 'lon']:
                if attr in data: data[attr] = float(data[attr])
        return G
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None

# ============================================================================
# NAVIGATION & SAFETY LOGIC
# ============================================================================

def get_node_coords(G, node):
    data = G.nodes[node]
    lat = data.get('y') or data.get('lat')
    lon = data.get('x') or data.get('lon')
    return float(lat), float(lon)

def find_nearest_node(G, target_lat, target_lon):
    min_dist = float('inf')
    nearest_node = None
    for node, data in G.nodes(data=True):
        lat, lon = get_node_coords(G, node)
        dist = calculate_haversine_distance(lat, lon, target_lat, target_lon)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node, min_dist

def calculate_route_safety(G, path):
    if len(path) < 2: return 'Safe', 0.1, 0, 'green'
    total_dist, weighted_safety = 0, 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            d = G[u][v]['raw_distance']
            total_dist += d
            weighted_safety += d * G[u][v]['safety_penalty']
    
    avg_p = weighted_safety / total_dist if total_dist > 0 else 1.0
    
    # Thresholds: Safe (p<0.3), Possible (0.3<=p<0.7), Hazardous (p>=0.7) [cite: 149, 179]
    if avg_p <= 1.2: return 'Safe', (avg_p-1.0)/2.0, total_dist, 'green'
    elif avg_p <= 2.0: return 'Possibly Hazardous', 0.3+(avg_p-1.2)*0.5, total_dist, 'orange'
    return 'Hazardous', min(0.7+(avg_p-2.0)*0.3, 1.0), total_dist, 'red'

def find_alternative_routes(G, start_node, end_node, k=3, algorithm='Dijkstra'):
    routes, G_temp = [], G.copy()
    for _ in range(k):
        start_t = time.perf_counter()
        try:
            if algorithm == 'A*':
                def h(u, v):
                    u_lat, u_lon = get_node_coords(G, u)
                    v_lat, v_lon = get_node_coords(G, v)
                    return calculate_haversine_distance(u_lat, u_lon, v_lat, v_lon)
                path = nx.astar_path(G_temp, start_node, end_node, heuristic=h, weight='weight')
            else:
                path = nx.dijkstra_path(G_temp, start_node, end_node, weight='weight')
            
            exec_ms = (time.perf_counter() - start_t) * 1000
            cls, prob, dist, color = calculate_route_safety(G, path)
            routes.append({
                'path': path, 'actual_distance': dist, 'safety_classification': cls, 
                'safety_probability': prob, 'color': color, 'exists': True, 
                'execution_time': exec_ms 
            })
            
            for j in range(len(path)-1):
                G_temp[path[j]][path[j+1]]['weight'] *= 10
        except nx.NetworkXNoPath: break
    return routes

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="Smart City Navigation", page_icon="🚗", layout="wide")
    st.title("🚗 Smart City Road Safety Navigation System")
    
    G = load_graph()
    df = pd.read_csv(Path(__file__).resolve().parent.parent / 'final_metadata.csv')
    
    for key in ['start_node', 'end_node', 'routes', 'hazard']:
        if key not in st.session_state: st.session_state[key] = None

    # Sidebar Controls
    st.sidebar.header("Navigation Controls")
    algorithm = st.sidebar.radio("Algorithm:", ["Dijkstra", "A*"], key="algo_selection")
    num_routes = st.sidebar.slider("Number of Alternatives", 1, 3, 3)
    
    # Auto-refresh when algorithm changes
    if st.session_state['start_node'] and st.session_state['end_node']:
        st.session_state['routes'] = find_alternative_routes(G, st.session_state['start_node'], st.session_state['end_node'], k=num_routes, algorithm=algorithm)

    if st.sidebar.button("Clear Selections", use_container_width=True):
        st.session_state.update({'start_node': None, 'end_node': None, 'routes': None, 'hazard': None})
        st.rerun()

    # Calculate Bounding Box
    lats = [float(data.get('y') or data.get('lat')) for _, data in G.nodes(data=True)]
    lons = [float(data.get('x') or data.get('lon')) for _, data in G.nodes(data=True)]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

    center_lat, center_lon = df.iloc[0]['latitude'], df.iloc[0]['longitude']
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 1. ALWAYS VISIBLE ROAD NETWORK
    road_layer = folium.FeatureGroup(name="Road Network")
    for u, v, data in list(G.edges(data=True))[:800]: 
        u_coords, v_coords = get_node_coords(G, u), get_node_coords(G, v)
        folium.PolyLine([u_coords, v_coords], color="darkblue", weight=2, opacity=0.4).add_to(road_layer)
    road_layer.add_to(m)

    # 2. SELECTION AREA BOX (FIXED FOR CLICKS)
    folium.Rectangle(
        bounds=bounds,
        color="red",
        weight=2,
        fill=True,
        fill_opacity=0.05,
        dash_array='5, 5',
        pointer_events=False  # THIS ALLOWS CLICKING THROUGH THE BOX
    ).add_to(m)

    # 3. FLOATING LEGEND
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; background-color: white; 
    border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; border-radius: 5px;">
    <b>Safety Legend</b><br>
    <i style="background:green; width:10px; height:10px; display:inline-block;"></i> Safe<br>
    <i style="background:orange; width:10px; height:10px; display:inline-block;"></i> Possible Hazard<br>
    <i style="background:red; width:10px; height:10px; display:inline-block;"></i> Hazardous
    </div>'''
    m.get_root().html.add_child(folium.Element(legend_html))

    # 4. ROUTE RENDERING & METRICS
    if st.session_state['routes']:
        selected_route_name = st.sidebar.selectbox("Active Route View:", [f"Route {i+1}" for i in range(len(st.session_state['routes']))])
        idx = int(selected_route_name.split()[-1]) - 1
        active_r = st.session_state['routes'][idx]

        for i in range(len(active_r['path']) - 1):
            u, v = active_r['path'][i], active_r['path'][i+1]
            u_coords, v_coords = get_node_coords(G, u), get_node_coords(G, v)
            penalty = G[u][v].get('safety_penalty', 1.0)
            color = 'green' if penalty <= 1.2 else 'orange' if penalty <= 2.0 else 'red'
            folium.PolyLine([u_coords, v_coords], color=color, weight=6, opacity=0.8).add_to(m)

        st.markdown(f"### 📊 Live Analysis: {selected_route_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Execution Time", f"{active_r['execution_time']:.3f} ms")
        c2.metric("Distance", f"{active_r['actual_distance']:.0f}m")
        c3.metric("Safety Score", f"{active_r['safety_probability']:.2f}")
        c4.metric("Status", active_r['safety_classification'])

    if st.session_state['start_node']:
        folium.Marker(get_node_coords(G, st.session_state['start_node']), icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
    if st.session_state['end_node']:
        folium.Marker(get_node_coords(G, st.session_state['end_node']), icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)

    map_data = st_folium(m, width=None, height=600, use_container_width=True, key="nav_map")
    
    # Click processing
    if map_data.get('last_clicked'):
        clicked_node, _ = find_nearest_node(G, map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
        if st.session_state['start_node'] is None:
            st.session_state['start_node'] = clicked_node
            st.rerun()
        elif st.session_state['end_node'] is None and clicked_node != st.session_state['start_node']:
            st.session_state['end_node'] = clicked_node
            st.rerun()

    st.markdown("---")
    if st.button("🎲 Simulate Random Hazard"):
        st.session_state['hazard'] = random.choice(list(G.edges()))
        st.warning(f"🚨 Hazard detected near node {st.session_state['hazard'][0]}. Safety re-routing advised.")
        st.rerun()

if __name__ == "__main__":
    main()