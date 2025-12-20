import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import osmnx as ox
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import cKDTree
from collections import defaultdict

# --- Configuration for Interactivity ---
BROWSER_MODE = False
try:
    matplotlib.use('WebAgg')
    BROWSER_MODE = True
except Exception:
    matplotlib.use('Agg')
    
# --- Safety and Weight Configuration ---
SAFETY_PENALTIES = {
    'safe': 1.0,
    'minor_issues': 1.5,
    'major_issues': 3.0,
    'major_problems': 3.0  
}

# --- Helper Functions ---

def calculate_haversine_distance(lat1, lon1, lat2, lon2, earth_radius_m=6371000):
    """Calculates the Haversine distance between two points (in meters)."""
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = earth_radius_m * c
    return distance

def get_bounding_box(nodes, padding=0.001):
    """Calculate bounding box from node coordinates with padding."""
    lats = [n['latitude'] for n in nodes]
    lons = [n['longitude'] for n in nodes]
    
    north = max(lats) + padding
    south = min(lats) - padding
    east = max(lons) + padding
    west = min(lons) - padding
    
    return north, south, east, west

def map_safety_data_to_edges(G, nodes, max_distance=50):
    """
    Maps safety ratings from collected nodes to OSM graph edges.
    Uses nearest neighbor matching within max_distance threshold.
    """
    print(f"\nMapping {len(nodes)} safety data points to OSM edges...")
    
    # Build spatial index of OSM edge midpoints
    edge_midpoints = []
    edge_list = []
    
    for u, v, key in G.edges(keys=True):
        # Get edge geometry
        if 'geometry' in G[u][v][key]:
            geom = G[u][v][key]['geometry']
            # Use midpoint of geometry
            coords = list(geom.coords)
            mid_idx = len(coords) // 2
            mid_lon, mid_lat = coords[mid_idx]
        else:
            # Use midpoint between nodes
            u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
            v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
            mid_lat = (u_lat + v_lat) / 2
            mid_lon = (u_lon + v_lon) / 2
        
        edge_midpoints.append([mid_lat, mid_lon])
        edge_list.append((u, v, key))
    
    # Build KD-tree for fast nearest neighbor search
    edge_tree = cKDTree(edge_midpoints)
    
    # Build KD-tree for safety nodes
    node_coords = [[n['latitude'], n['longitude']] for n in nodes]
    node_tree = cKDTree(node_coords)
    
    # Map each safety node to nearest edge
    edge_safety_data = defaultdict(list)
    matched_count = 0
    
    for i, node in enumerate(nodes):
        node_coord = [node['latitude'], node['longitude']]
        
        # Find nearest edge
        dist, edge_idx = edge_tree.query(node_coord)
        
        # Convert to meters (KD-tree returns in coordinate units)
        dist_meters = calculate_haversine_distance(
            node_coord[0], node_coord[1],
            edge_midpoints[edge_idx][0], edge_midpoints[edge_idx][1]
        )
        
        if dist_meters <= max_distance:
            u, v, key = edge_list[edge_idx]
            edge_safety_data[(u, v, key)].append({
                'condition': node['road_condition'],
                'distance': dist_meters
            })
            matched_count += 1
    
    print(f"  • Matched {matched_count}/{len(nodes)} data points to edges")
    print(f"  • {len(edge_safety_data)} unique edges have safety data")
    
    # Aggregate safety data for each edge (use worst condition)
    condition_priority = {
        'safe': 0,
        'minor_issues': 1,
        'major_issues': 2,
        'major_problems': 2
    }
    
    for (u, v, key), data_points in edge_safety_data.items():
        # Find worst condition
        worst_condition = 'safe'
        worst_priority = -1
        
        for point in data_points:
            condition = point['condition']
            priority = condition_priority.get(condition, 0)
            if priority > worst_priority:
                worst_priority = priority
                worst_condition = condition
        
        # Update edge attributes
        penalty = SAFETY_PENALTIES.get(worst_condition, 1.0)
        length = G[u][v][key].get('length', 100)  # OSM provides length in meters
        
        G[u][v][key]['road_condition'] = worst_condition
        G[u][v][key]['safety_penalty'] = penalty
        G[u][v][key]['weight'] = length * penalty
        G[u][v][key]['has_safety_data'] = True
    
    # Set default values for edges without safety data
    edges_without_data = 0
    for u, v, key in G.edges(keys=True):
        if 'has_safety_data' not in G[u][v][key]:
            length = G[u][v][key].get('length', 100)
            G[u][v][key]['road_condition'] = 'safe'
            G[u][v][key]['safety_penalty'] = 1.0
            G[u][v][key]['weight'] = length * 1.0
            G[u][v][key]['has_safety_data'] = False
            edges_without_data += 1
    
    print(f"  • {edges_without_data} edges without data (defaulted to 'safe')")
    
    return G

def build_osm_road_network(nodes, network_type='drive', custom_filter=None):
    """
    Downloads OSM road network for the area covered by nodes.
    
    Args:
        nodes: List of node dictionaries with lat/lon
        network_type: OSM network type ('drive', 'walk', 'bike', 'all')
        custom_filter: Custom OSM filter (optional)
    """
    print("\n" + "=" * 60)
    print("Building OSM-Based Road Network Graph")
    print("=" * 60)
    
    # Calculate bounding box
    north, south, east, west = get_bounding_box(nodes)
    bbox = (west, south, east, north)
    
    print(f"\nBounding Box:")
    print(f"  • North: {north:.6f}")
    print(f"  • South: {south:.6f}")
    print(f"  • East:  {east:.6f}")
    print(f"  • West:  {west:.6f}")
    
    # Download OSM graph
    print(f"\nDownloading OSM road network (type: {network_type})...")
    try:
        if custom_filter:
            G = ox.graph_from_bbox(bbox=bbox, 
                                   custom_filter=custom_filter, 
                                   simplify=True)
        else:
            G = ox.graph_from_bbox(bbox=bbox, 
                                   network_type=network_type, 
                                   simplify=True)
    except Exception as e:
        print(f"✗ Error downloading OSM data: {e}")
        print("\nTrying with smaller bounding box...")
        
        # Try with smaller padding
        n2, s2, e2, w2 = get_bounding_box(nodes, padding=0.002)
        bbox_small = (w2, s2, e2, n2)
        G = ox.graph_from_bbox(bbox=bbox_small, 
                               network_type=network_type, 
                               simplify=True)
    
    # Convert to MultiDiGraph if not already
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    
    return G

def print_graph_statistics(G):
    """Prints statistics about the graph."""
    print("\nGraph Statistics:")
    print(f"  • Nodes: {G.number_of_nodes()}")
    print(f"  • Edges: {G.number_of_edges()}")
    
    # Check connectivity
    G_undirected = G.to_undirected()
    is_connected = nx.is_connected(G_undirected)
    print(f"  • Connected: {is_connected}")
    
    if not is_connected:
        num_components = nx.number_connected_components(G_undirected)
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        print(f"  • Number of components: {num_components}")
        print(f"  • Largest component size: {len(largest_cc)} nodes")
    
    # Safety data statistics
    edges_with_data = sum(1 for u, v, k, d in G.edges(keys=True, data=True) 
                         if d.get('has_safety_data', False))
    print(f"  • Edges with safety data: {edges_with_data}/{G.number_of_edges()}")
    
    # Condition distribution
    conditions = [d.get('road_condition', 'safe') 
                 for u, v, k, d in G.edges(keys=True, data=True)]
    condition_counts = pd.Series(conditions).value_counts()
    print(f"\nRoad Condition Distribution:")
    for condition, count in condition_counts.items():
        print(f"  • {condition}: {count}")

def visualize_graph(G, nodes, filename='osm_road_network_with_safety.png'):
    """Visualize the OSM graph with safety overlay."""
    global BROWSER_MODE
    
    condition_to_color = {
        'safe': 'green',
        'minor_issues': 'gold',
        'major_issues': 'red',
        'major_problems': 'red'
    }
    
    print("\nGenerating visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Plot base OSM graph (light gray for roads without data)
    edge_colors = []
    edge_widths = []
    
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get('has_safety_data', False):
            condition = data.get('road_condition', 'safe')
            edge_colors.append(condition_to_color.get(condition, 'gray'))
            edge_widths.append(3)
        else:
            edge_colors.append('lightgray')
            edge_widths.append(1)
    
    # Draw graph
    ox.plot_graph(G, ax=ax, node_size=20, node_color='black', 
                  edge_color=edge_colors, edge_linewidth=edge_widths,
                  bgcolor='white', show=False, close=False)
    
    # Overlay original data collection points
    lats = [n['latitude'] for n in nodes]
    lons = [n['longitude'] for n in nodes]
    node_colors = [condition_to_color.get(n['road_condition'], 'gray') 
                   for n in nodes]
    
    ax.scatter(lons, lats, c=node_colors, s=50, alpha=0.7, 
              edgecolors='black', linewidths=0.5, zorder=5,
              label='Data Collection Points')
    
    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color='green', linewidth=3, label='Safe'),
        plt.Line2D([0], [0], color='gold', linewidth=3, label='Minor Issues'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Major Issues/Problems'),
        plt.Line2D([0], [0], color='lightgray', linewidth=1, label='No Data (Default Safe)'),
        plt.Line2D([0], [0], marker='o', color='w', label='Collection Points',
                  markerfacecolor='black', markersize=8)
    ]
    ax.legend(handles=legend_handles, title="Road Safety", loc='upper left', fontsize=10)
    
    ax.set_title(f"OSM Road Network with Safety Data\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if BROWSER_MODE:
        plt.show()
    else:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {filename}")
        plt.close()

def save_graph(G, filename='osm_road_network.graphml'):
    """Save graph to file for later use."""
    print(f"\nSaving graph to {filename}...")
    ox.save_graphml(G, filepath=filename)
    print(f"✓ Graph saved successfully")

def load_graph(filename='osm_road_network.graphml'):
    """Load graph from file."""
    print(f"\nLoading graph from {filename}...")
    G = ox.load_graphml(filename)
    print(f"✓ Graph loaded successfully")
    return G

# --- Main Execution Logic ---

def main():
    print("=" * 60)
    print("OSM-Based Road Network Graph Builder")
    print("=" * 60)
    print("\nLoading data from final_metadata.csv...")
    
    try:
        df = pd.read_csv('final_metadata.csv')
    except FileNotFoundError:
        print("✗ Error: final_metadata.csv not found.")
        return
    
    # Prepare nodes
    nodes = df.apply(lambda row: {
        'original_index': row['index'] if 'index' in row else row.name,
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'road_condition': row['road_condition'] if row['road_condition'] != 'none' else 'safe'
    }, axis=1).tolist()
    
    print(f"✓ Loaded {len(nodes)} data points")
    
    if len(nodes) < 2:
        print("\n✗ Not enough nodes to build a network.")
        return
    
    # Build OSM road network
    G = build_osm_road_network(nodes, network_type='drive')
    
    # Map safety data to edges
    G = map_safety_data_to_edges(G, nodes, max_distance=50)
    
    # Print statistics
    print_graph_statistics(G)
    
    # Visualize
    visualize_graph(G, nodes)
    
    # Save graph
    save_graph(G, 'osm_road_network.graphml')
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    print("\nYou can now use this graph for pathfinding with:")
    print("  - nx.shortest_path(G, source, target, weight='weight')")
    print("  - nx.dijkstra_path(G, source, target, weight='weight')")
    print("  - nx.astar_path(G, source, target, weight='weight')")

if __name__ == "__main__":
    main()