import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics.pairwise import haversine_distances
from collections import Counter
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
    # MODIFICATION 1: Removed 'none' as it's now mapped to 'safe' in pre-processing
    'minor_issues': 1.5,
    'major_issues': 3.0,
    'major_problems': 3.0  
}

# --- Data Pre-processing Function ---
def normalize_road_conditions(nodes):
    """
    Replaces all 'none' road conditions with 'safe'.
    """
    for node in nodes:
        if node['road_condition'] == 'none':
            node['road_condition'] = 'safe'
    return nodes
# --- Graph Building Functions ---

def calculate_haversine_distance(lat1, lon1, lat2, lon2, earth_radius_m=6371000):
    """
    Calculates the Haversine distance between two points (in meters).
    """
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

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing (direction) between two points in degrees."""
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    bearing = np.arctan2(delta_lon, delta_lat)
    return np.degrees(bearing)

def is_collinear(p1, p2, p3, angle_threshold=15):
    """Check if three points are approximately collinear."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    lat3, lon3 = p3
    
    bearing1 = calculate_bearing(lat1, lon1, lat2, lon2)
    bearing2 = calculate_bearing(lat2, lon2, lat3, lon3)
    
    angle_diff = abs(bearing1 - bearing2)
    
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    return angle_diff < angle_threshold

def average_nearby_nodes(nodes, min_dist_threshold=25):
    """
    Groups consecutive nodes that are within a close distance threshold.
    Averages their coordinates and takes the majority vote for road condition.
    """
    if not nodes:
        return []

    averaged_nodes = []
    
    # Initialize the first batch
    batch = [nodes[0]]
    current_batch_dist = 0
    
    for i in range(1, len(nodes)):
        curr = nodes[i]
        prev = nodes[i-1]
        
        # Calculate distance from previous node
        d = calculate_haversine_distance(prev['latitude'], prev['longitude'], 
                                         curr['latitude'], curr['longitude'])
        current_batch_dist += d
        
        # If we are within the threshold, add to batch
        if current_batch_dist < min_dist_threshold:
            batch.append(curr)
        else:
            # Batch is full (distance exceeded) -> Process it
            # 1. Average Position
            avg_lat = np.mean([n['latitude'] for n in batch])
            avg_lon = np.mean([n['longitude'] for n in batch])
            
            # 2. Majority Vote Condition
            conditions = [n['road_condition'] for n in batch]
            most_common_condition = Counter(conditions).most_common(1)[0][0]
            
            new_node = {
                'original_index': batch[0]['original_index'], 
                'latitude': avg_lat,
                'longitude': avg_lon,
                'road_condition': most_common_condition
            }
            averaged_nodes.append(new_node)
            
            # Start new batch with current node
            batch = [curr]
            current_batch_dist = 0
            
    # Process any remaining nodes in the final batch
    if batch:
        avg_lat = np.mean([n['latitude'] for n in batch])
        avg_lon = np.mean([n['longitude'] for n in batch])
        conditions = [n['road_condition'] for n in batch]
        most_common_condition = Counter(conditions).most_common(1)[0][0]
        
        new_node = {
            'original_index': batch[0]['original_index'],
            'latitude': avg_lat,
            'longitude': avg_lon,
            'road_condition': most_common_condition
        }
        averaged_nodes.append(new_node)
        
    return averaged_nodes

def reduce_collinear_nodes(nodes, angle_threshold=15):
    """
    Reduce nodes by removing intermediate points that lie on straight segments.
    Condition check is removed to allow longer, single-color edges.
    """
    if len(nodes) <= 2:
        return nodes
    
    reduced_nodes = [nodes[0]]
    
    for i in range(1, len(nodes) - 1):
        prev_node = reduced_nodes[-1]
        curr_node = nodes[i]
        next_node = nodes[i + 1]
        
        # Keep node if it's a turning point (not collinear)
        prev_pos = (prev_node['latitude'], prev_node['longitude'])
        curr_pos = (curr_node['latitude'], curr_node['longitude'])
        next_pos = (next_node['latitude'], next_node['longitude'])
        
        if not is_collinear(prev_pos, curr_pos, next_pos, angle_threshold):
            reduced_nodes.append(curr_node)
    
    reduced_nodes.append(nodes[-1])
    
    return reduced_nodes

def build_weighted_road_network_graph(nodes, simplify=True, angle_threshold=15, averaging_threshold=25):
    """
    Builds a CONNECTED bidirectional, weighted road network graph.
    """
    original_count = len(nodes)
    
    # STEP 1: Spatial Averaging (Smoothing) 
    nodes = average_nearby_nodes(nodes, min_dist_threshold=averaging_threshold)
    print(f"  • Nodes averaged (smoothed) from {original_count} to {len(nodes)} (Threshold: {averaging_threshold}m)")
    
    # STEP 2: Collinear Reduction (Geometry optimization)
    if simplify:
        intermediate_count = len(nodes)
        nodes = reduce_collinear_nodes(nodes, angle_threshold)
        print(f"  • Nodes reduced (geometry) from {intermediate_count} to {len(nodes)}")
    
    G = nx.DiGraph()
    
    for idx, node_data in enumerate(nodes):
        G.add_node(idx, 
                   original_index=node_data['original_index'],
                   pos=(node_data['longitude'], node_data['latitude']),
                   road_condition=node_data['road_condition'])
                   
    for i in range(len(nodes) - 1):
        source_data = nodes[i]
        source_idx = i
        target_data = nodes[i+1]
        target_idx = i+1
        
        # The weight calculation remains the same
        distance = calculate_haversine_distance(
            source_data['latitude'], source_data['longitude'],
            target_data['latitude'], target_data['longitude']
        )
        
        # The edge condition/penalty is based on the source node 
        condition = source_data['road_condition']
        penalty = SAFETY_PENALTIES.get(condition, 5.0)
        weighted_cost = distance * penalty
        
        G.add_edge(source_idx, target_idx, 
                   weight=weighted_cost, 
                   raw_distance=distance,
                   safety_penalty=penalty,
                   condition=condition) # Added condition to edge for visualization
                   
        G.add_edge(target_idx, source_idx, 
                   weight=weighted_cost, 
                   raw_distance=distance,
                   safety_penalty=penalty,
                   condition=condition)
            
    return G

def print_graph_statistics(G):
    """Prints basic statistics about the graph."""
    print("\nGraph Statistics:")
    print(f"  • Nodes: {G.number_of_nodes()}")
    print(f"  • Edges (Segments x 2): {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        G_undirected = G.to_undirected()
        is_connected = nx.is_connected(G_undirected)
        print(f"  • Connected: {is_connected}")
        
        if not is_connected:
            num_components = nx.number_connected_components(G_undirected)
            print(f"  • Number of components: {num_components}")
    
    if G.number_of_nodes() > 1:
        density = nx.density(G)
        edges = G.edges(data=True)
        avg_cost = np.mean([d['weight'] for _, _, d in edges])
        avg_dist = np.mean([d['raw_distance'] for _, _, d in edges])
        print(f"  • Average edge cost: {avg_cost:.6f}")
        print(f"  • Average edge distance (m): {avg_dist:.6f}")

def visualize_graph(G, filename='road_network_graph_weighted.png'):
    global BROWSER_MODE
    # MODIFICATION 2: Removed 'none' as it's now 'safe'
    condition_to_color = {
        'minor_issues': 'gold',
        'major_issues': 'red',
        'major_problems': 'red',
        'safe': 'green' 
    }
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get node colors
    node_conditions = nx.get_node_attributes(G, 'road_condition').values()
    node_colors = [condition_to_color.get(c, 'red') for c in node_conditions]
    
    # MODIFICATION 3: Get edge colors based on the 'condition' attribute we added
    edge_conditions = nx.get_edge_attributes(G, 'condition')
    # Filter to unique edges (u < v) for consistent coloring
    edge_color_map = {}
    for (u, v), condition in edge_conditions.items():
        if u < v:
            edge_color_map[(u, v)] = condition_to_color.get(condition, 'gray')

    # Create a list of colors in the order of the edges in the graph
    # This requires iterating through G.edges to maintain order
    edge_colors = [edge_color_map.get((u, v) if u < v else (v, u), 'gray') for u, v, _ in G.edges(data=True)]

    plt.figure(figsize=(14, 14))
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.9)
    # Use the calculated edge_colors array
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.9, width=2.5, 
                          arrows=True, arrowsize=10, arrowstyle='->')
    
    edge_weights = nx.get_edge_attributes(G, 'weight')
    unique_weights = {}
    for (u, v), weight in edge_weights.items():
        if u < v:
            unique_weights[(u, v)] = f"{weight:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=unique_weights, font_size=7, label_pos=0.5)

    # MODIFICATION 4: Update legend handles (removed 'none' entry)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Safe', markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Minor Issues', markerfacecolor='gold', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Major Issues/Problems', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_handles, title="Road Condition", loc='upper left')

    plt.title(f"Simplified Road Network Graph - {G.number_of_nodes()} Key Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if BROWSER_MODE:
        plt.show() 
    else:
        plt.savefig(filename, dpi=150)
        print(f"\nStatic visualization saved to {filename}")
        plt.close()

# --- Main Execution Logic ---

def main():
    print("=" * 60)
    print("Connected Road Network Graph Builder (Final Simplified)")
    print("=" * 60)
    print("\nLoading data from final_metadata.csv...")
    
    try:
        df = pd.read_csv('final_metadata.csv')
    except FileNotFoundError:
        print("Error: final_metadata.csv not found.")
        return
    
    if 'index' in df.columns:
        df = df.sort_values('index').reset_index(drop=True)
    
    nodes = df.apply(lambda row: {
        'original_index': row['index'] if 'index' in row else row.name,
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'road_condition': row['road_condition']
    }, axis=1).tolist()
    
    print(f"\n  • Original points: {len(nodes)}")

    # MODIFICATION 5: Pre-process nodes to convert 'none' to 'safe'
    nodes = normalize_road_conditions(nodes)
    
    if len(nodes) < 2:
        print("\nNot enough nodes to build a network.")
        return
    
    NEW_AVERAGING_THRESHOLD = 75 
    
    print(f"\nBuilding graph (Averaging: {NEW_AVERAGING_THRESHOLD}m, 'none' -> 'safe')...")
    
    G = build_weighted_road_network_graph(nodes, 
                                          simplify=True, 
                                          angle_threshold=15,
                                          averaging_threshold=NEW_AVERAGING_THRESHOLD)
    
    print(f"\n✓ Graph built successfully!")
    print_graph_statistics(G)
    
    print("\nGenerating visualization...")
    visualize_graph(G, filename='road_network_graph_final_colored.png')
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()