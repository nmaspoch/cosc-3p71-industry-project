import streamlit as st
import pickle
import networkx as nx
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('..')
from component2.graph import calculate_haversine_distance

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Smart City Navigation System",
    page_icon="🚗",
    layout="wide"
)

# ============================================================================
# LOAD DATA (Cached for performance)
# ============================================================================

@st.cache_resource
def load_graph():
    """Load the YOLO-enhanced graph"""
    graph_path = Path('data/graph_with_yolo.gpickle')
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G

@st.cache_data
def load_metadata():
    """Load the original metadata"""
    return pd.read_csv('../final_metadata.csv')

# ============================================================================
# NAVIGATION FUNCTIONS
# ============================================================================

def find_nearest_node(G, target_lat, target_lon):
    """Find the closest node in graph to given coordinates"""
    min_dist = float('inf')
    nearest_node = None
    
    for node, data in G.nodes(data=True):
        lon, lat = data['pos']
        dist = calculate_haversine_distance(lat, lon, target_lat, target_lon)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node, min_dist

def calculate_route_safety(G, path):
    """Calculate overall safety classification for a route"""
    if len(path) < 2:
        return 'safe', 0.1, 0
    
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
    
    # Calculate average penalty
    avg_penalty = weighted_safety / total_distance if total_distance > 0 else 1.0
    
    # Convert to probability scale and classify
    if avg_penalty <= 1.2:
        classification = 'Safe'
        probability = (avg_penalty - 1.0) / 2.0  # 0.0 - 0.1
        color = 'green'
    elif avg_penalty <= 2.0:
        classification = 'Possibly Hazardous'
        probability = 0.3 + (avg_penalty - 1.2) * 0.5  # 0.3 - 0.7
        color = 'orange'
    else:
        classification = 'Hazardous'
        probability = min(0.7 + (avg_penalty - 2.0) * 0.3, 1.0)  # 0.7 - 1.0
        color = 'red'
    
    return classification, probability, total_distance, color

def find_route(G, start_node, end_node):
    """Find the shortest weighted path using Dijkstra's algorithm"""
    try:
        path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
        path_length = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')
        
        classification, probability, actual_distance, color = calculate_route_safety(G, path)
        
        return {
            'path': path,
            'weighted_cost': path_length,
            'actual_distance': actual_distance,
            'safety_classification': classification,
            'safety_probability': probability,
            'color': color,
            'exists': True
        }
    except nx.NetworkXNoPath:
        return {'exists': False}

def find_alternative_routes(G, start_node, end_node, k=3):
    """Find k alternative routes by temporarily removing edges"""
    routes = []
    G_temp = G.copy()
    
    for i in range(k):
        route = find_route(G_temp, start_node, end_node)
        
        if not route['exists']:
            break
        
        routes.append(route)
        
        # Remove edges from found path to force alternatives
        path = route['path']
        for j in range(len(path) - 1):
            if G_temp.has_edge(path[j], path[j+1]):
                G_temp.remove_edge(path[j], path[j+1])
            if G_temp.has_edge(path[j+1], path[j]):
                G_temp.remove_edge(path[j+1], path[j])
    
    return routes

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_base_folium_map(G, center_lat=None, center_lon=None):
    """Create base Folium map with road network"""
    # Calculate center if not provided
    if center_lat is None or center_lon is None:
        lats = [data['pos'][1] for _, data in G.nodes(data=True)]
        lons = [data['pos'][0] for _, data in G.nodes(data=True)]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 450px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; padding: 10px">
    <b>Smart City Road Safety Navigation System</b><br>
    Component 4: Intelligent Route Planning with YOLO Safety Scores
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def add_graph_edges_to_map(m, G, show_all_edges=False):
    """Add graph edges to map with safety color coding"""
    
    condition_colors = {
        'safe': 'green',
        'minor_issues': 'gold',
        'major_issues': 'red',
        'major_problems': 'red'
    }
    
    edges_added = set()
    
    for u, v, data in G.edges(data=True):
        # Avoid duplicate lines (since graph is bidirectional)
        edge_key = tuple(sorted([u, v]))
        if edge_key in edges_added:
            continue
        edges_added.add(edge_key)
        
        # Get node positions
        u_lon, u_lat = G.nodes[u]['pos']
        v_lon, v_lat = G.nodes[v]['pos']
        
        # Get edge condition
        condition = data.get('condition', 'safe')
        color = condition_colors.get(condition, 'gray')
        
        # Create tooltip with edge info
        tooltip = f"Condition: {condition}<br>Distance: {data['raw_distance']:.0f}m<br>Safety Penalty: {data['safety_penalty']}x"
        
        if show_all_edges:
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=color,
                weight=2,
                opacity=0.4,
                tooltip=tooltip
            ).add_to(m)

def add_route_to_map(m, G, route, label="Route"):
    """Add a calculated route to the map"""
    path_coords = []
    
    for node in route['path']:
        lon, lat = G.nodes[node]['pos']
        path_coords.append([lat, lon])
    
    # Create popup with route stats
    popup_html = f"""
    <div style="width: 200px">
        <b>{label}</b><br>
        <b>Classification:</b> {route['safety_classification']}<br>
        <b>Distance:</b> {route['actual_distance']:.0f}m<br>
        <b>Safety Score:</b> {route['safety_probability']:.3f}<br>
        <b>Segments:</b> {len(route['path']) - 1}
    </div>
    """
    
    folium.PolyLine(
        path_coords,
        color=route['color'],
        weight=6,
        opacity=0.8,
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(m)
    
    return path_coords

def add_markers_to_map(m, start_coords, end_coords):
    """Add start and end markers"""
    folium.Marker(
        start_coords,
        popup="Start Point",
        tooltip="Start",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        end_coords,
        popup="End Point",
        tooltip="End",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)

def add_legend_to_map(m):
    """Add route safety legend"""
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>Route Safety Legend</b><br>
    <i style="color:green">━━━</i> Safe (p < 0.3)<br>
    <i style="color:orange">━━━</i> Possibly Hazardous (0.3-0.7)<br>
    <i style="color:red">━━━</i> Hazardous (p ≥ 0.7)<br>
    <br>
    <b>Road Conditions:</b><br>
    <span style="color:green">●</span> Safe  
    <span style="color:gold">●</span> Minor Issues  
    <span style="color:red">●</span> Major Issues
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("🚗 Smart City Road Safety Navigation System")
    st.markdown("### Component 4: Intelligent Route Planning with AI-Enhanced Safety Scores")
    
    # Load data
    with st.spinner("Loading navigation system..."):
        G = load_graph()
        df = load_metadata()
    
    st.success(f"✓ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # ========================================================================
    # SIDEBAR: Input Controls
    # ========================================================================
    
    st.sidebar.header("Navigation Controls")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["Preset Locations", "Manual Coordinates", "Random Points"]
    )
    
    if input_method == "Preset Locations":
        # Create preset interesting locations from your data
        locations = {}
        for idx in [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]:
            row = df.iloc[idx]
            locations[f"Point {idx} ({row['road_condition']})"] = (row['latitude'], row['longitude'])
        
        start_loc = st.sidebar.selectbox("Start Location:", list(locations.keys()), index=0)
        end_loc = st.sidebar.selectbox("End Location:", list(locations.keys()), index=len(locations)-1)
        
        start_lat, start_lon = locations[start_loc]
        end_lat, end_lon = locations[end_loc]
        
    elif input_method == "Manual Coordinates":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown("**Start Point**")
            start_lat = st.number_input("Latitude", value=df['latitude'].iloc[0], format="%.6f", key="start_lat")
            start_lon = st.number_input("Longitude", value=df['longitude'].iloc[0], format="%.6f", key="start_lon")
        with col2:
            st.markdown("**End Point**")
            end_lat = st.number_input("Latitude", value=df['latitude'].iloc[-1], format="%.6f", key="end_lat")
            end_lon = st.number_input("Longitude", value=df['longitude'].iloc[-1], format="%.6f", key="end_lon")
    
    else:  # Random Points
        if st.sidebar.button("Generate Random Points"):
            random_indices = np.random.choice(len(df), 2, replace=False)
            st.session_state['random_start'] = random_indices[0]
            st.session_state['random_end'] = random_indices[1]
        
        start_idx = st.session_state.get('random_start', 0)
        end_idx = st.session_state.get('random_end', len(df)-1)
        
        start_lat, start_lon = df.iloc[start_idx]['latitude'], df.iloc[start_idx]['longitude']
        end_lat, end_lon = df.iloc[end_idx]['latitude'], df.iloc[end_idx]['longitude']
        
        st.sidebar.info(f"Start: Index {start_idx}\nEnd: Index {end_idx}")
    
    # Display options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    show_road_network = st.sidebar.checkbox("Show Road Network", value=False)
    num_routes = st.sidebar.slider("Number of Alternative Routes", 1, 3, 3)
    
    # Calculate button
    calculate_routes = st.sidebar.button("🔍 Calculate Routes", type="primary")
    
    # ========================================================================
    # MAIN CONTENT: Route Calculation and Visualization
    # ========================================================================
    
    if calculate_routes:
        with st.spinner("Finding nearest nodes..."):
            start_node, start_dist = find_nearest_node(G, start_lat, start_lon)
            end_node, end_dist = find_nearest_node(G, end_lat, end_lon)
        
        st.info(f"📍 Start node: {start_node} (Distance: {start_dist:.1f}m from input)\n\n"
                f"📍 End node: {end_node} (Distance: {end_dist:.1f}m from input)")
        
        if start_node == end_node:
            st.error("Start and end points are too close! Please select different locations.")
            return
        
        with st.spinner(f"Calculating {num_routes} alternative routes..."):
            routes = find_alternative_routes(G, start_node, end_node, k=num_routes)
        
        if not routes:
            st.error("❌ No routes found between selected points!")
            return
        
        st.success(f"✓ Found {len(routes)} route(s)")
        
        # ====================================================================
        # Display Route Statistics
        # ====================================================================
        
        st.markdown("---")
        st.subheader("📊 Route Comparison")
        
        cols = st.columns(len(routes))
        for i, (route, col) in enumerate(zip(routes, cols)):
            with col:
                # Color-coded metric based on safety
                if route['safety_classification'] == 'Safe':
                    icon = "✅"
                elif route['safety_classification'] == 'Possibly Hazardous':
                    icon = "⚠️"
                else:
                    icon = "🚨"
                
                st.markdown(f"### Route {i+1} {icon}")
                st.metric("Distance", f"{route['actual_distance']:.0f}m")
                st.metric("Safety Classification", route['safety_classification'])
                st.metric("Safety Probability", f"{route['safety_probability']:.3f}")
                st.metric("Segments", len(route['path']) - 1)
        
        # ====================================================================
        # Create and Display Map
        # ====================================================================
        
        st.markdown("---")
        st.subheader("🗺️ Navigation Map")
        
        # Create base map
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        m = create_base_folium_map(G, center_lat, center_lon)
        
        # Add road network if requested
        if show_road_network:
            add_graph_edges_to_map(m, G, show_all_edges=True)
        
        # Add all routes
        for i, route in enumerate(routes):
            add_route_to_map(m, G, route, label=f"Route {i+1}")
        
        # Add start/end markers
        start_coords_actual = [G.nodes[start_node]['pos'][1], G.nodes[start_node]['pos'][0]]
        end_coords_actual = [G.nodes[end_node]['pos'][1], G.nodes[end_node]['pos'][0]]
        add_markers_to_map(m, start_coords_actual, end_coords_actual)
        
        # Add legend
        add_legend_to_map(m)
        
        # Display map
        st_folium(m, width=1200, height=600)
        
        # ====================================================================
        # Detailed Route Information
        # ====================================================================
        
        with st.expander("📋 Detailed Route Information"):
            for i, route in enumerate(routes):
                st.markdown(f"#### Route {i+1} Details")
                
                # Create dataframe of edges
                edge_data = []
                for j in range(len(route['path']) - 1):
                    u, v = route['path'][j], route['path'][j+1]
                    edge = G[u][v]
                    edge_data.append({
                        'Segment': j+1,
                        'From Node': u,
                        'To Node': v,
                        'Distance (m)': f"{edge['raw_distance']:.1f}",
                        'Condition': edge['condition'],
                        'Safety Penalty': f"{edge['safety_penalty']}x",
                        'YOLO Probability': f"{edge.get('yolo_probability', 'N/A')}"
                    })
                
                st.dataframe(pd.DataFrame(edge_data), use_container_width=True)
                st.markdown("---")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()