import streamlit as st
import pickle
import networkx as nx
import folium
import osmnx as ox  # Added for OSMnx 2.0+ compatibility
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import random

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from component2.graph import calculate_haversine_distance

def get_coords(G, node_id):
    """Returns (lon, lat) for a given node, compatible with both OSMnx and legacy formats."""
    node_data = G.nodes[node_id]
    if 'pos' in node_data:
        return node_data['pos']  # (lon, lat)
    # OSMnx standard: x is longitude, y is latitude
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
# LOAD DATA (Updated for OSMnx 2.0+ and GraphML)
# ============================================================================

@st.cache_resource
def load_graph():
    """Load the graph and normalize attributes for the navigation app."""
    graph_path = Path('osm_road_network.graphml')
    
    if not graph_path.exists():
        st.error("❌ 'osm_road_network.graphml' not found. Please run graph.py first.")
        return None

    try:
        # 1. Load the graph
        G_raw = ox.load_graphml(filepath=str(graph_path))
        
        # 2. Convert MultiDiGraph to DiGraph (picks the shortest edge between nodes)
        # This fixes the structure so G[u][v] works correctly in your app
        G = ox.convert.to_digraph(G_raw, weight='weight')
        
        # 3. Normalize attributes and ensure numeric types
        for u, v, data in G.edges(data=True):
            # Rename 'length' to 'raw_distance' to match your app's logic
            if 'length' in data:
                data['raw_distance'] = float(data['length'])
            elif 'raw_distance' in data:
                data['raw_distance'] = float(data['raw_distance'])
            else:
                data['raw_distance'] = 100.0 # Fallback
                
            # Rename 'road_condition' to 'condition'
            if 'road_condition' in data:
                data['condition'] = data['road_condition']
            
            # Ensure other pathfinding weights are floats
            for attr in ['weight', 'safety_penalty', 'yolo_probability']:
                if attr in data:
                    try:
                        data[attr] = float(data[attr])
                    except (ValueError, TypeError):
                        data[attr] = 1.0

        # 4. Ensure node coordinates are floats
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
# NAVIGATION FUNCTIONS (Updated for x/y coordinate handling)
# ============================================================================

def get_node_coords(G, node):
    """Safely get lat/lon regardless of attribute names."""
    data = G.nodes[node]
    # OSMnx standard is 'y' for lat, 'x' for lon
    lat = data.get('y') or data.get('lat')
    lon = data.get('x') or data.get('lon')
    return float(lon), float(lat)

def find_nearest_node(G, target_lat, target_lon):
    """Find the closest node in graph to given coordinates"""
    min_dist = float('inf')
    nearest_node = None
    
    for node, data in G.nodes(data=True):
        # Support both standard OSMnx (x,y) and custom (pos) formats
        lon, lat = get_node_coords(G, node)
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

def calculate_travel_time(G, path):
    """
    Calculate estimated travel time for a route based on distance and road conditions.
    Returns time in minutes.
    """
    if len(path) < 2:
        return 0.0
    
    total_time_hours = 0.0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            edge_data = G[u][v]
            distance_km = edge_data['raw_distance'] / 1000  # Convert m to km
            condition = edge_data.get('condition', 'safe')
            speed = SPEED_CONFIG.get(condition, DEFAULT_SPEED)
            
            # Time = Distance / Speed
            segment_time = distance_km / speed
            total_time_hours += segment_time
    
    # Convert hours to minutes
    return total_time_hours * 60

def find_route(G, start_node, end_node, algorithm='dijkstra'):
    """Find the shortest weighted path using specified algorithm with travel time"""
    try:
        if algorithm == 'astar':
            # A* requires a heuristic function
            def heuristic(u, v):
                """Heuristic: straight-line distance between nodes"""
                # Use the get_coords helper to handle x/y or pos keys
                u_lon, u_lat = get_coords(G, u)
                v_lon, v_lat = get_coords(G, v)
                return calculate_haversine_distance(u_lat, u_lon, v_lat, v_lon)
            
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight='weight')
            path_length = nx.astar_path_length(G, start_node, end_node, heuristic=heuristic, weight='weight')
        else:  # dijkstra
            path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
            path_length = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')
        
        classification, probability, actual_distance, color = calculate_route_safety(G, path)
        travel_time = calculate_travel_time(G, path)
        
        return {
            'path': path,
            'weighted_cost': path_length,
            'actual_distance': actual_distance,
            'travel_time': travel_time,
            'safety_classification': classification,
            'safety_probability': probability,
            'color': color,
            'exists': True
        }
    except nx.NetworkXNoPath:
        return {'exists': False}

def find_alternative_routes(G, start_node, end_node, k=3, algorithm='dijkstra'):
    """Find k alternative routes by penalizing used edges instead of removing them"""
    routes = []
    G_temp = G.copy()
    
    for i in range(k):
        route = find_route(G_temp, start_node, end_node, algorithm=algorithm)
        
        if not route['exists']:
            break
        
        routes.append(route)
        
        # Instead of removing edges, heavily penalize them to find alternatives
        # This works better for linear/loop graphs
        path = route['path']
        for j in range(len(path) - 1):
            u, v = path[j], path[j+1]
            if G_temp.has_edge(u, v):
                # Multiply weight by 10 to discourage reuse
                G_temp[u][v]['weight'] = G_temp[u][v]['weight'] * 10
            if G_temp.has_edge(v, u):
                G_temp[v][u]['weight'] = G_temp[v][u]['weight'] * 10
    
    return routes

# ============================================================================
# REAL-TIME DETECTION FUNCTIONS
# ============================================================================

def simulate_random_hazard(G, excluded_edges=None):
    """
    Simulate detection of a new road hazard on a random edge.
    Returns the affected edge (u, v) and new condition.
    """
    if excluded_edges is None:
        excluded_edges = set()
    
    # Get all edges not in excluded set
    available_edges = [(u, v) for u, v, _ in G.edges(data=True) 
                       if (u, v) not in excluded_edges]
    
    if not available_edges:
        return None
    
    # Select random edge
    u, v = random.choice(available_edges)
    
    # Simulate hazard detection
    hazard_types = ['major_issues', 'major_problems']
    new_condition = random.choice(hazard_types)
    
    # Calculate new safety penalty
    if new_condition in ['major_issues', 'major_problems']:
        new_penalty = 3.0
    else:
        new_penalty = 2.0
    
    return {
        'edge': (u, v),
        'new_condition': new_condition,
        'old_condition': G[u][v].get('condition', 'safe'),
        'new_penalty': new_penalty,
        'old_penalty': G[u][v]['safety_penalty'],
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

def apply_hazard_to_graph(G, hazard_info):
    """
    Apply a detected hazard to the graph by updating edge weights.
    Returns modified graph.
    """
    G_updated = G.copy()
    u, v = hazard_info['edge']
    
    # Update both directions if edge exists
    if G_updated.has_edge(u, v):
        G_updated[u][v]['condition'] = hazard_info['new_condition']
        G_updated[u][v]['safety_penalty'] = hazard_info['new_penalty']
        G_updated[u][v]['weight'] = (G_updated[u][v]['raw_distance'] * 
                                     hazard_info['new_penalty'])
    
    if G_updated.has_edge(v, u):
        G_updated[v][u]['condition'] = hazard_info['new_condition']
        G_updated[v][u]['safety_penalty'] = hazard_info['new_penalty']
        G_updated[v][u]['weight'] = (G_updated[v][u]['raw_distance'] * 
                                     hazard_info['new_penalty'])
    
    return G_updated

def check_route_affected(route, hazard_info):
    """Check if a route is affected by a detected hazard"""
    u, v = hazard_info['edge']
    path = route['path']
    
    for i in range(len(path) - 1):
        if (path[i] == u and path[i+1] == v) or (path[i] == v and path[i+1] == u):
            return True
    return False

def calculate_route_comparison(original_route, new_route):
    """Compare original route with re-routed path"""
    time_diff = new_route['travel_time'] - original_route['travel_time']
    dist_diff = new_route['actual_distance'] - original_route['actual_distance']
    
    return {
        'time_difference': time_diff,
        'distance_difference': dist_diff,
        'is_faster': time_diff < 0,
        'is_shorter': dist_diff < 0,
        'safety_improved': (new_route['safety_probability'] < 
                          original_route['safety_probability'])
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_base_folium_map(G, center_lat=None, center_lon=None):
    """Create base Folium map with road network"""
    if center_lat is None or center_lon is None:
        coords = [get_node_coords(G, n) for n in G.nodes()]
        lons, lats = zip(*coords)
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    # ... [title HTML remains same] ...
    return m

def add_graph_edges_to_map(m, G, show_all_edges=False):
    """Add graph edges to map with safety color coding"""
    condition_colors = {'safe': 'green', 'minor_issues': 'gold', 'major_issues': 'red', 'major_problems': 'red'}
    edges_added = set()
    
    for u, v, data in G.edges(data=True):
        edge_key = tuple(sorted([u, v]))
        if edge_key in edges_added: continue
        edges_added.add(edge_key)
        
        u_lon, u_lat = get_node_coords(G, u)
        v_lon, v_lat = get_node_coords(G, v)
        
        condition = data.get('condition', 'safe')
        color = condition_colors.get(condition, 'gray')
        tooltip = f"Condition: {condition}<br>Distance: {data['raw_distance']:.0f}m"
        
        if show_all_edges:
            folium.PolyLine([[u_lat, u_lon], [v_lat, v_lon]], color=color, weight=2, opacity=0.4, tooltip=tooltip).add_to(m)

def add_route_to_map(m, G, route, label="Route", route_color=None):
    """Add a calculated route to the map"""
    path_coords = []
    for node in route['path']:
        lon, lat = get_node_coords(G, node)
        path_coords.append([lat, lon])
    
    route_color = route_color or route['color']
    popup_html = f"<b>{label}</b><br>Time: {route['travel_time']:.1f} min"
    
    folium.PolyLine(path_coords, color=route_color, weight=6, opacity=0.8, popup=folium.Popup(popup_html)).add_to(m)
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
        ["Preset Locations", "Select Nodes", "Random Points"]
    )
    
    if input_method == "Preset Locations":
        # Better preset locations spread across the loop
        locations = {}
        indices = [0, 15, 30, 45, 60]  # Spread around the loop
        for idx in indices:
            if idx < len(df):
                row = df.iloc[idx]
                locations[f"Point {idx} ({row['road_condition']})"] = (row['latitude'], row['longitude'])
        
        start_loc = st.sidebar.selectbox("Start Location:", list(locations.keys()), index=0)
        end_loc = st.sidebar.selectbox("End Location:", list(locations.keys()), index=2)  # Changed default
        
        start_lat, start_lon = locations[start_loc]
        end_lat, end_lon = locations[end_loc]
        
    elif input_method == "Select Nodes":
        st.sidebar.markdown("**Select Graph Nodes**")
        
        # Get all nodes and their positions
        all_nodes = list(G.nodes())
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_node_select = st.selectbox("Start Node:", all_nodes, index=0, key="start_node_select")
        with col2:
            end_node_select = st.selectbox("End Node:", all_nodes, index=min(30, len(all_nodes)-1), key="end_node_select")
        
        # Get coordinates from selected nodes
        start_lon, start_lat = G.nodes[start_node_select]['pos']
        end_lon, end_lat = G.nodes[end_node_select]['pos']
        
        st.sidebar.info(f"Start Node {start_node_select}: ({start_lat:.5f}, {start_lon:.5f})\n\n"
                       f"End Node {end_node_select}: ({end_lat:.5f}, {end_lon:.5f})")
    
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
    
    # Algorithm selection
    st.sidebar.markdown("**Pathfinding Algorithm**")
    algorithm = st.sidebar.radio(
        "Select Algorithm:",
        ["dijkstra", "astar"],
        format_func=lambda x: "Dijkstra's Algorithm" if x == "dijkstra" else "A* Algorithm"
    )
    
    # Route visibility toggles
    st.sidebar.markdown("**Route Visibility**")
    route_visibility = {}
    for i in range(3):
        route_visibility[i] = st.sidebar.checkbox(f"Show Route {i+1}", value=True, key=f"route_viz_{i}")
    
    # Calculate button
    if st.sidebar.button("🔍 Calculate Routes", type="primary"):
        st.session_state['should_calculate'] = True
        
    # ========================================================================
    # MAIN CONTENT: Route Calculation and Visualization
    # ========================================================================
    
    if st.session_state.get('should_calculate', False):
        with st.spinner("Finding nearest nodes..."):
            start_node, start_dist = find_nearest_node(G, start_lat, start_lon)
            end_node, end_dist = find_nearest_node(G, end_lat, end_lon)
        
        st.info(f"🎯 Start node: {start_node} (Distance: {start_dist:.1f}m from input)\n\n"
                f"🎯 End node: {end_node} (Distance: {end_dist:.1f}m from input)")
        
        if start_node == end_node:
            st.error("Start and end points are too close! Please select different locations.")
        else:
            with st.spinner(f"Calculating {num_routes} alternative routes using {algorithm.upper()}..."):
                routes = find_alternative_routes(G, start_node, end_node, k=num_routes, algorithm=algorithm)
            
            if not routes:
                st.error("❌ No routes found between selected points!")
            else:
                # Show route count and info
                algorithm_name = "Dijkstra's Algorithm" if algorithm == "dijkstra" else "A* Algorithm"
                st.success(f"✓ Found {len(routes)} route(s) using {algorithm_name}")
                
                if len(routes) == 1:
                    st.info("ℹ️ Only one route available. Your road network may have limited alternative paths between these points.")
                
                # Display Route Statistics (WITH TRAVEL TIME)
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
                        st.metric("Travel Time", f"{route['travel_time']:.1f} min")
                        st.metric("Safety Classification", route['safety_classification'])
                        st.metric("Safety Probability", f"{route['safety_probability']:.3f}")
                        st.metric("Segments", len(route['path']) - 1)
                
                # Create and Display Map
                st.markdown("---")
                st.subheader("🗺️ Navigation Map")
                
                # Create base map
                center_lat = (start_lat + end_lat) / 2
                center_lon = (start_lon + end_lon) / 2
                m = create_base_folium_map(G, center_lat, center_lon)
                
                # Add road network if requested
                if show_road_network:
                    add_graph_edges_to_map(m, G, show_all_edges=True)
                
                # Add all routes with DISTINCT colors
                # Replace the route drawing loop in your main() with this:

            for i, route in enumerate(routes):
                # Only add route if visibility is enabled
                if not route_visibility.get(i, False):
                    continue
                
                path = route['path']
                # Iterate through segments to color them individually
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    
                    # Get coordinates for this segment
                    u_lon, u_lat = get_coords(G, u)
                    v_lon, v_lat = get_coords(G, v)
                    segment_coords = [[u_lat, u_lon], [v_lat, v_lon]]
                    
                    # Determine segment color based on the edge data
                    edge_data = G[u][v]
                    penalty = edge_data.get('safety_penalty', 1.0)
                    
                    if penalty <= 1.2:
                        seg_color = 'green'
                    elif penalty <= 2.0:
                        seg_color = 'orange' # Yellow/Orange
                    else:
                        seg_color = 'red'
                        
                    # Add the individual segment to the map
                    folium.PolyLine(
                        segment_coords,
                        color=seg_color,
                        weight=8 - (i * 2), # Maintain thickness difference for overlapping routes
                        opacity=0.9,
                        tooltip=f"Route {i+1} Segment: {edge_data.get('condition', 'safe')}"
                    ).add_to(m)
                
                # Add start/end markers
                # Note: Folium uses [lat, lon], while get_coords returns (lon, lat)
                lon_s, lat_s = get_coords(G, start_node)
                lon_e, lat_e = get_coords(G, end_node)
                start_coords_actual = [lat_s, lon_s]
                end_coords_actual = [lat_e, lon_e]
                add_markers_to_map(m, start_coords_actual, end_coords_actual)
                
                # Add legend with route colors (only show visible routes)
                visible_routes = [i for i in range(len(routes)) if route_visibility.get(i, False)]
                legend_lines = []
                colors = ['blue', 'purple', 'darkred']
                for i in visible_routes:
                    color = colors[i] if i < len(colors) else 'gray'
                    legend_lines.append(f'<span style="color:{color}; font-size:20px">━━</span> Route {i+1}<br>')
                
                legend_html = f'''
                <div style="position: fixed; 
                            bottom: 50px; right: 50px; width: 200px; height: auto; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px">
                <b>Visible Routes: {len(visible_routes)}</b><br>
                {''.join(legend_lines)}
                <br>
                <b>Road Conditions:</b><br>
                <span style="color:green">●</span> Safe  
                <span style="color:gold">●</span> Minor Issues  
                <span style="color:red">●</span> Major Issues
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
                # Display map
                st_folium(m, width=1200, height=600)
                
                # ============================================================
                # REAL-TIME HAZARD DETECTION & RE-ROUTING
                # ============================================================
                
                st.markdown("---")
                st.subheader("🚨 Real-Time Hazard Detection & Re-routing")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("""
                    This simulation demonstrates the system's ability to:
                    - Detect new road hazards in real-time
                    - Automatically re-route traffic around problems
                    - Notify users of changes and alternatives
                    """)
                
                with col2:
                    if st.button("🎲 Simulate Hazard Detection", type="secondary"):
                        st.session_state['trigger_hazard'] = True
                
                # Process hazard detection if triggered
                if st.session_state.get('trigger_hazard', False):
                    st.session_state['trigger_hazard'] = False
                    
                    with st.spinner("🔍 Detecting hazard..."):
                        # Simulate hazard detection
                        hazard = simulate_random_hazard(G)
                        
                        if hazard:
                            st.session_state['detected_hazards'] = st.session_state.get('detected_hazards', [])
                            st.session_state['detected_hazards'].append(hazard)
                            
                            # Check which routes are affected
                            affected_routes = []
                            for i, route in enumerate(routes):
                                if check_route_affected(route, hazard):
                                    affected_routes.append(i)
                            
                            # Store hazard info
                            st.session_state['last_hazard'] = hazard
                            st.session_state['affected_routes'] = affected_routes
                            
                            # Apply hazard to graph
                            st.session_state['G_updated'] = apply_hazard_to_graph(G, hazard)
                            
                            # Store original routes for comparison
                            st.session_state['original_routes'] = routes
                            st.session_state['start_node'] = start_node
                            st.session_state['end_node'] = end_node
                
                # Display Notification System
                if st.session_state.get('last_hazard'):
                    hazard = st.session_state['last_hazard']
                    affected = st.session_state.get('affected_routes', [])
                    
                    # Notification Panel
                    if affected:
                        st.error(f"""
                        ⚠️ **HAZARD DETECTED** at {hazard['timestamp']}
                        
                        **Location:** Between nodes {hazard['edge'][0]} and {hazard['edge'][1]}
                        
                        **Type:** {hazard['new_condition'].replace('_', ' ').title()}
                        
                        **Affected Routes:** Route {', Route '.join([str(i+1) for i in affected])}
                        
                        **Action Required:** Re-routing recommended
                        """)
                        
                        # Calculate alternative routes
                        if st.button("🔄 Calculate Alternative Routes", type="primary"):
                            G_updated = st.session_state['G_updated']
                            original_routes = st.session_state.get('original_routes', routes)
                            
                            with st.spinner("Finding safer alternatives..."):
                                new_routes = find_alternative_routes(
                                    G_updated, start_node, end_node, k=num_routes, algorithm=algorithm
                                )
                                
                                if new_routes:
                                    st.success(f"✅ Found {len(new_routes)} alternative route(s)")
                                    
                                    # Compare with original routes
                                    st.markdown("### 📊 Route Comparison: Original vs. Alternative")
                                    
                                    for i in affected:
                                        if i < len(original_routes) and i < len(new_routes):
                                            original = original_routes[i]
                                            alternative = new_routes[i]
                                            comparison = calculate_route_comparison(original, alternative)
                                            
                                            st.markdown(f"#### Route {i+1} Analysis")
                                            col1, col2, col3 = st.columns(3)
                                            
                                            with col1:
                                                st.metric(
                                                    "Time Change",
                                                    f"{alternative['travel_time']:.1f} min",
                                                    f"{comparison['time_difference']:+.1f} min"
                                                )
                                            
                                            with col2:
                                                st.metric(
                                                    "Distance Change",
                                                    f"{alternative['actual_distance']:.0f}m",
                                                    f"{comparison['distance_difference']:+.0f}m"
                                                )
                                            
                                            with col3:
                                                safety_change = (alternative['safety_probability'] - 
                                                               original['safety_probability'])
                                                st.metric(
                                                    "Safety Change",
                                                    f"{alternative['safety_probability']:.3f}",
                                                    f"{safety_change:+.3f}"
                                                )
                                            
                                            # Recommendation
                                            if comparison['safety_improved']:
                                                st.success(f"✅ **Recommendation:** Take alternative route - Safer path available")
                                            elif comparison['is_faster']:
                                                st.info(f"ℹ️ **Recommendation:** Alternative route is {abs(comparison['time_difference']):.1f} min faster")
                                            else:
                                                st.warning(f"⚠️ **Recommendation:** Original route affected - Consider alternatives")
                                            
                                            st.markdown("---")
                                    
                                    # Store new routes for visualization
                                    st.session_state['alternative_routes'] = new_routes
                    else:
                        st.info(f"""
                        ℹ️ **Hazard Detected** at {hazard['timestamp']}
                        
                        **Location:** Between nodes {hazard['edge'][0]} and {hazard['edge'][1]}
                        
                        **Status:** Your current routes are not affected by this hazard.
                        """)
                
                # Display all detected hazards history
                if st.session_state.get('detected_hazards'):
                    with st.expander("📋 Hazard Detection History"):
                        hazards_df = pd.DataFrame([
                            {
                                'Time': h['timestamp'],
                                'Edge': f"{h['edge'][0]} → {h['edge'][1]}",
                                'Type': h['new_condition'].replace('_', ' ').title(),
                                'Old Condition': h['old_condition'],
                                'Penalty Change': f"{h['old_penalty']}x → {h['new_penalty']}x"
                            }
                            for h in st.session_state['detected_hazards']
                        ])
                        st.dataframe(hazards_df, use_container_width=True)
                        
                        if st.button("🗑️ Clear History"):
                            st.session_state['detected_hazards'] = []
                            st.session_state['last_hazard'] = None
                            st.rerun()
                
                # Detailed Route Information
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