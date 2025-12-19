import pickle
import networkx as nx
import folium
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import from graph.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from component2.graph import update_edge_safety_from_yolo, calculate_haversine_distance

# ============================================================================
# LOAD DATA
# ============================================================================

def load_graph():
    """Load the YOLO-enhanced graph"""
    graph_path = Path('data/graph_with_yolo.gpickle')
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G

def load_metadata():
    """Load the original metadata"""
    return pd.read_csv('../final_metadata.csv')

# ============================================================================
# NAVIGATION FUNCTIONS (Same as navigation_app.py)
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
        return 'Safe', (avg_penalty - 1.0) / 2.0, total_distance, 'green'
    elif avg_penalty <= 2.0:
        return 'Possibly Hazardous', 0.3 + (avg_penalty - 1.2) * 0.5, total_distance, 'orange'
    else:
        return 'Hazardous', min(0.7 + (avg_penalty - 2.0) * 0.3, 1.0), total_distance, 'red'

def find_route(G, start_node, end_node):
    """Find the shortest weighted path"""
    try:
        path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
        classification, probability, distance, color = calculate_route_safety(G, path)
        
        return {
            'path': path,
            'actual_distance': distance,
            'safety_classification': classification,
            'safety_probability': probability,
            'color': color,
            'exists': True
        }
    except nx.NetworkXNoPath:
        return {'exists': False}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_map(G, original_route, new_route, hazard_edge, center_lat, center_lon):
    """Create a map showing before/after routes"""
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 500px; height: 80px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; padding: 10px">
    <b>Dynamic Re-routing Demonstration</b><br>
    <span style="color:blue">━━━</span> Original Route<br>
    <span style="color:purple">━━━</span> Re-routed Path (After Hazard Detection)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add original route in blue
    original_coords = []
    for node in original_route['path']:
        lon, lat = G.nodes[node]['pos']
        original_coords.append([lat, lon])
    
    folium.PolyLine(
        original_coords,
        color='blue',
        weight=5,
        opacity=0.7,
        popup=f"Original Route<br>Distance: {original_route['actual_distance']:.0f}m<br>Safety: {original_route['safety_classification']}",
        dash_array='5, 5'
    ).add_to(m)
    
    # Add new route in purple
    new_coords = []
    for node in new_route['path']:
        lon, lat = G.nodes[node]['pos']
        new_coords.append([lat, lon])
    
    folium.PolyLine(
        new_coords,
        color='purple',
        weight=6,
        opacity=0.9,
        popup=f"New Route (After Re-routing)<br>Distance: {new_route['actual_distance']:.0f}m<br>Safety: {new_route['safety_classification']}"
    ).add_to(m)
    
    # Highlight the hazardous edge
    u, v = hazard_edge
    u_lon, u_lat = G.nodes[u]['pos']
    v_lon, v_lat = G.nodes[v]['pos']
    
    folium.PolyLine(
        [[u_lat, u_lon], [v_lat, v_lon]],
        color='red',
        weight=8,
        opacity=1.0,
        popup="🚨 NEW HAZARD DETECTED",
        dash_array='10, 5'
    ).add_to(m)
    
    # Add hazard marker
    mid_lat = (u_lat + v_lat) / 2
    mid_lon = (u_lon + v_lon) / 2
    folium.Marker(
        [mid_lat, mid_lon],
        popup="⚠️ Hazard Detected Here",
        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
    ).add_to(m)
    
    # Add start/end markers
    start_lat, start_lon = G.nodes[original_route['path'][0]]['pos'][1], G.nodes[original_route['path'][0]]['pos'][0]
    end_lat, end_lon = G.nodes[original_route['path'][-1]]['pos'][1], G.nodes[original_route['path'][-1]]['pos'][0]
    
    folium.Marker([start_lat, start_lon], popup="Start", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
    folium.Marker([end_lat, end_lon], popup="End", icon=folium.Icon(color='darkred', icon='stop', prefix='fa')).add_to(m)
    
    return m

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def run_rerouting_demo():
    print("=" * 80)
    print("DYNAMIC RE-ROUTING DEMONSTRATION")
    print("Component 4: Real-Time Adaptation to Road Conditions")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading navigation system...")
    G = load_graph()
    df = load_metadata()
    print(f"    ✓ Loaded graph with {G.number_of_nodes()} nodes")
    
    # Select start and end points
    print("\n[2] Selecting navigation points...")
    start_idx = 10
    end_idx = len(df) - 20
    
    start_lat, start_lon = df.iloc[start_idx]['latitude'], df.iloc[start_idx]['longitude']
    end_lat, end_lon = df.iloc[end_idx]['latitude'], df.iloc[end_idx]['longitude']
    
    start_node, _ = find_nearest_node(G, start_lat, start_lon)
    end_node, _ = find_nearest_node(G, end_lat, end_lon)
    
    print(f"    ✓ Start Node: {start_node}")
    print(f"    ✓ End Node: {end_node}")
    
    # Calculate original route
    print("\n[3] Calculating original route...")
    original_route = find_route(G, start_node, end_node)
    
    if not original_route['exists']:
        print("    ✗ No route found! Try different points.")
        return
    
    print(f"    ✓ Route found!")
    print(f"      • Path length: {len(original_route['path'])} nodes")
    print(f"      • Distance: {original_route['actual_distance']:.0f}m")
    print(f"      • Safety: {original_route['safety_classification']}")
    print(f"      • Probability: {original_route['safety_probability']:.3f}")
    
    # Simulate hazard detection on route
    print("\n[4] Simulating hazard detection...")
    
    # Pick an edge in the middle of the route to make hazardous
    hazard_position = len(original_route['path']) // 2
    hazard_edge = (original_route['path'][hazard_position], 
                   original_route['path'][hazard_position + 1])
    
    print(f"    🚨 NEW HAZARD DETECTED on edge {hazard_edge}")
    print(f"       Updating safety score to 0.90 (hazardous)")
    
    # Get current edge info
    current_condition = G[hazard_edge[0]][hazard_edge[1]]['condition']
    current_prob = G[hazard_edge[0]][hazard_edge[1]].get('yolo_probability', 'N/A')
    
    print(f"       Previous condition: {current_condition}")
    print(f"       Previous probability: {current_prob}")
    
    # Update graph with new hazard
    update_edge_safety_from_yolo(G, {hazard_edge: 0.90})
    
    new_condition = G[hazard_edge[0]][hazard_edge[1]]['condition']
    new_prob = G[hazard_edge[0]][hazard_edge[1]]['yolo_probability']
    
    print(f"       New condition: {new_condition}")
    print(f"       New probability: {new_prob}")
    
    # Recalculate route
    print("\n[5] Recalculating route with updated conditions...")
    new_route = find_route(G, start_node, end_node)
    
    if not new_route['exists']:
        print("    ✗ No alternative route found!")
        return
    
    print(f"    ✓ New route calculated!")
    print(f"      • Path length: {len(new_route['path'])} nodes")
    print(f"      • Distance: {new_route['actual_distance']:.0f}m")
    print(f"      • Safety: {new_route['safety_classification']}")
    print(f"      • Probability: {new_route['safety_probability']:.3f}")
    
    # Compare routes
    print("\n[6] Route Comparison:")
    print("    " + "=" * 60)
    print(f"    {'Metric':<25} {'Original':<15} {'After Re-routing':<15}")
    print("    " + "-" * 60)
    print(f"    {'Distance (m)':<25} {original_route['actual_distance']:<15.0f} {new_route['actual_distance']:<15.0f}")
    print(f"    {'Safety Classification':<25} {original_route['safety_classification']:<15} {new_route['safety_classification']:<15}")
    print(f"    {'Safety Probability':<25} {original_route['safety_probability']:<15.3f} {new_route['safety_probability']:<15.3f}")
    print(f"    {'Number of Segments':<25} {len(original_route['path'])-1:<15} {len(new_route['path'])-1:<15}")
    
    # Check if route actually changed
    if original_route['path'] != new_route['path']:
        print("\n    ✅ SUCCESS: Route was re-calculated to avoid the hazard!")
        distance_change = new_route['actual_distance'] - original_route['actual_distance']
        if distance_change > 0:
            print(f"    📏 Additional distance: +{distance_change:.0f}m ({(distance_change/original_route['actual_distance']*100):.1f}% increase)")
        print(f"    🛡️  Safety improved from {original_route['safety_probability']:.3f} to {new_route['safety_probability']:.3f}")
    else:
        print("\n    ⚠️  Route unchanged: No alternative path available")
    
    print("    " + "=" * 60)
    
    # Create visualization
    print("\n[7] Creating visualization...")
    center_lat = (start_lat + end_lat) / 2
    center_lon = (start_lon + end_lon) / 2
    
    m = create_comparison_map(G, original_route, new_route, hazard_edge, center_lat, center_lon)
    
    output_file = 'dynamic_rerouting_demo.html'
    m.save(output_file)
    
    print(f"    ✓ Map saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("\nThis demonstrates the system's ability to:")
    print("  • Detect new road hazards in real-time")
    print("  • Update route safety assessments dynamically")
    print("  • Recalculate optimal routes to avoid dangers")
    print("  • Provide safer alternatives to users")
    print("\nOpen 'dynamic_rerouting_demo.html' to view the visualization.")
    print("=" * 80)

if __name__ == "__main__":
    run_rerouting_demo()