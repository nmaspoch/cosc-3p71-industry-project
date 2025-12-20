import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import sys
from pathlib import Path
from ultralytics import YOLO
import osmnx as ox
import networkx as nx
from scipy.spatial import cKDTree

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import OSM graph building functions
from component2.graph import build_osm_road_network, map_safety_data_to_edges

def calculate_haversine_distance(lat1, lon1, lat2, lon2, earth_radius_m=6371000):
    """Calculate Haversine distance between two points (in meters)."""
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

def get_edge_geometry_points(G, u, v, key):
    """
    Get points along an edge for distance calculation.
    Returns list of (lat, lon) tuples.
    """
    if 'geometry' in G[u][v][key]:
        # Edge has geometry - use all points
        geom = G[u][v][key]['geometry']
        coords = list(geom.coords)
        # Convert from (lon, lat) to (lat, lon)
        return [(lat, lon) for lon, lat in coords]
    else:
        # No geometry - use start and end nodes only
        u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
        v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
        return [(u_lat, u_lon), (v_lat, v_lon)]

def point_to_edge_distance(point_lat, point_lon, edge_points):
    """
    Calculate minimum distance from a point to an edge (represented by multiple points).
    Returns distance in meters.
    """
    min_dist = float('inf')
    
    # Check distance to each segment of the edge
    for i in range(len(edge_points) - 1):
        lat1, lon1 = edge_points[i]
        lat2, lon2 = edge_points[i + 1]
        
        dist = point_to_line_segment_distance(
            point_lat, point_lon,
            lat1, lon1,
            lat2, lon2
        )
        min_dist = min(min_dist, dist)
    
    return min_dist

def point_to_line_segment_distance(point_lat, point_lon, line_start_lat, line_start_lon, 
                                   line_end_lat, line_end_lon):
    """
    Calculate perpendicular distance from a point to a line segment.
    Returns distance in meters.
    """
    px, py = point_lon, point_lat
    ax, ay = line_start_lon, line_start_lat
    bx, by = line_end_lon, line_end_lat
    
    AB_x = bx - ax
    AB_y = by - ay
    
    AP_x = px - ax
    AP_y = py - ay
    
    AB_AB = AB_x * AB_x + AB_y * AB_y
    AP_AB = AP_x * AB_x + AP_y * AB_y
    
    if AB_AB == 0:
        return calculate_haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)
    
    t = max(0, min(1, AP_AB / AB_AB))
    
    closest_x = ax + t * AB_x
    closest_y = ay + t * AB_y
    
    return calculate_haversine_distance(point_lat, point_lon, closest_y, closest_x)

def map_images_to_edges(G, images_df, max_distance=50):
    """
    Map each image to its nearest OSM graph edge.
    Returns: dict mapping (u, v, key) edge tuples to lists of image indices
    
    Args:
        G: OSM MultiDiGraph
        images_df: DataFrame with image metadata
        max_distance: Maximum distance (meters) to consider for mapping
    """
    print("\n" + "="*60)
    print("MAPPING IMAGES TO OSM GRAPH EDGES")
    print("="*60)
    
    edge_to_images = {}
    
    # Get all edges (OSM uses MultiDiGraph with keys)
    edges = list(G.edges(keys=True))
    
    # Initialize edge mapping
    for u, v, key in edges:
        edge_to_images[(u, v, key)] = []
    
    print(f"\nProcessing {len(images_df)} images against {len(edges)} edges...")
    
    # Build spatial index for faster lookup
    # Get edge midpoints for initial filtering
    edge_midpoints = []
    edge_list = []
    
    for u, v, key in edges:
        points = get_edge_geometry_points(G, u, v, key)
        mid_idx = len(points) // 2
        mid_lat, mid_lon = points[mid_idx]
        edge_midpoints.append([mid_lat, mid_lon])
        edge_list.append((u, v, key))
    
    edge_tree = cKDTree(edge_midpoints)
    
    # Process each image
    for idx, row in images_df.iterrows():
        img_lat = row['latitude']
        img_lon = row['longitude']
        img_index = row['index'] if 'index' in row else idx
        
        # Find candidate edges using spatial index (rough filter)
        # Search for edges within 2x max_distance in coordinate space
        coord_dist = max_distance / 111000  # Rough conversion to degrees
        indices = edge_tree.query_ball_point([img_lat, img_lon], coord_dist * 2)
        
        if not indices:
            # No nearby edges found
            continue
        
        min_distance = float('inf')
        nearest_edge = None
        
        # Check actual distance to candidate edges
        for edge_idx in indices:
            u, v, key = edge_list[edge_idx]
            edge_points = get_edge_geometry_points(G, u, v, key)
            
            dist = point_to_edge_distance(img_lat, img_lon, edge_points)
            
            if dist < min_distance:
                min_distance = dist
                nearest_edge = (u, v, key)
        
        # Assign image to nearest edge if within threshold
        if nearest_edge and min_distance <= max_distance:
            edge_to_images[nearest_edge].append(img_index)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(images_df)} images...")
    
    # Print statistics
    total_mapped = sum(len(imgs) for imgs in edge_to_images.values())
    edges_with_images = sum(1 for imgs in edge_to_images.values() if len(imgs) > 0)
    
    print(f"\n✓ Mapping complete!")
    print(f"  • Total images mapped: {total_mapped}/{len(images_df)}")
    print(f"  • Edges with images: {edges_with_images}/{len(edges)}")
    if edges_with_images > 0:
        print(f"  • Avg images per edge: {total_mapped/edges_with_images:.1f}")
    
    return edge_to_images

def load_yolo_model(model_path='yolov8n.pt'):
    """Load YOLO model for inference."""
    print(f"\nLoading YOLO model from {model_path}...")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
    return model

def run_yolo_inference_single(model, image_path):
    """
    Run YOLO inference on a single image.
    Returns probability score (0-1) for hazard detection.
    """
    try:
        results = model(image_path, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            max_idx = np.argmax(confidences)
            max_conf = float(confidences[max_idx])
            predicted_class = int(classes[max_idx])
            
            # Map class to probability based on severity
            class_to_probability = {
                0: 0.1,   # safe
                1: 0.5,   # minor_issues
                2: 0.8,   # major_issues
                3: 0.9    # major_problems
            }
            
            base_prob = class_to_probability.get(predicted_class, 0.5)
            return base_prob * max_conf
        else:
            return 0.1
            
    except Exception as e:
        print(f"  Warning: Error processing {image_path}: {e}")
        return 0.2

def calculate_edge_safety_scores(G, edge_to_images, model, images_df):
    """
    Run YOLO inference on all images and calculate safety score per edge.
    """
    print("\n" + "="*60)
    print("CALCULATING EDGE SAFETY SCORES WITH YOLO")
    print("="*60)
    
    edge_safety_scores = {}
    total_edges = len(edge_to_images)
    processed = 0
    
    for edge_tuple, image_indices in edge_to_images.items():
        processed += 1
        
        if not image_indices:
            edge_safety_scores[edge_tuple] = 0.2
            continue
        
        probabilities = []
        
        for img_idx in image_indices:
            img_row = images_df[images_df['index'] == img_idx]
            
            if len(img_row) == 0:
                continue
            
            img_filename = img_row.iloc[0]['filename']
            img_path = Path('dataset/images/all') / img_filename
            
            if img_path.exists():
                prob = run_yolo_inference_single(model, str(img_path))
                probabilities.append(prob)
            else:
                print(f"  Warning: Image not found: {img_path}")
        
        if probabilities:
            # Use worst case (max) for safety-critical application
            max_prob = max(probabilities)
            edge_safety_scores[edge_tuple] = max_prob
            
            if processed % 50 == 0:
                print(f"  [{processed}/{total_edges}] Edge {edge_tuple}: "
                      f"{len(probabilities)} images, max_prob={max_prob:.3f}")
        else:
            edge_safety_scores[edge_tuple] = 0.2
    
    print(f"\n✓ Calculated safety scores for {len(edge_safety_scores)} edges")
    
    # Print distribution
    scores = [s for s in edge_safety_scores.values() if s > 0]
    if scores:
        print(f"\nScore Distribution:")
        print(f"  • Min:  {min(scores):.3f}")
        print(f"  • Mean: {np.mean(scores):.3f}")
        print(f"  • Max:  {max(scores):.3f}")
        print(f"  • Safe (p<0.3):           {sum(1 for s in scores if s < 0.3)}")
        print(f"  • Possibly Hazardous:     {sum(1 for s in scores if 0.3 <= s < 0.7)}")
        print(f"  • Hazardous (p>=0.7):     {sum(1 for s in scores if s >= 0.7)}")
    
    return edge_safety_scores

def update_graph_with_yolo_scores(G, edge_safety_scores):
    """
    Update OSM graph edges with YOLO-based safety scores.
    """
    print("\nUpdating graph with YOLO-based weights...")
    
    updated_count = 0
    
    for (u, v, key), prob_score in edge_safety_scores.items():
        if G.has_edge(u, v, key):
            # Determine condition based on probability
            if prob_score < 0.3:
                condition = 'safe'
                penalty = 1.0
            elif prob_score < 0.7:
                condition = 'minor_issues'
                penalty = 1.5
            else:
                condition = 'major_issues'
                penalty = 3.0
            
            length = G[u][v][key].get('length', 100)
            new_weight = length * penalty
            
            G[u][v][key]['road_condition'] = condition
            G[u][v][key]['safety_penalty'] = penalty
            G[u][v][key]['weight'] = new_weight
            G[u][v][key]['yolo_probability'] = prob_score
            G[u][v][key]['has_yolo_data'] = True
            
            updated_count += 1
    
    print(f"✓ Updated {updated_count} edges with YOLO scores")
    
    return G

def save_outputs(G, edge_to_images, edge_safety_scores):
    """Save all outputs to files."""
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    output_dir = Path('component4/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save graph using OSMnx format
    graph_base_path = output_dir / 'osm_graph_base.graphml'
    ox.save_graphml(G, graph_base_path)
    print(f"✓ Base OSM graph saved to: {graph_base_path}")
    
    # Save edge-to-images mapping
    mapping_path = output_dir / 'edge_to_images.json'
    mapping_json = {str(k): v for k, v in edge_to_images.items()}
    with open(mapping_path, 'w') as f:
        json.dump(mapping_json, f, indent=2)
    print(f"✓ Edge mapping saved to: {mapping_path}")
    
    # Save safety scores
    scores_path = output_dir / 'edge_safety_scores.json'
    scores_json = {str(k): v for k, v in edge_safety_scores.items()}
    with open(scores_path, 'w') as f:
        json.dump(scores_json, f, indent=2)
    print(f"✓ Safety scores saved to: {scores_path}")
    
    # Update graph with YOLO scores
    G = update_graph_with_yolo_scores(G, edge_safety_scores)
    
    # Save YOLO-enhanced graph
    graph_yolo_path = output_dir / 'osm_graph_with_yolo.graphml'
    ox.save_graphml(G, graph_yolo_path)
    print(f"✓ YOLO-enhanced graph saved to: {graph_yolo_path}")
    
    # Also save as pickle for faster loading
    graph_pickle_path = output_dir / 'osm_graph_with_yolo.gpickle'
    with open(graph_pickle_path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"✓ Pickle version saved to: {graph_pickle_path}")

def main():
    print("="*60)
    print("COMPONENT 4: EDGE SAFETY CALCULATION WITH YOLO (OSM)")
    print("="*60)
    
    # Step 1: Load metadata
    print("\nStep 1: Loading metadata...")
    try:
        df = pd.read_csv('final_metadata.csv')
        print(f"✓ Loaded {len(df)} image records")
    except FileNotFoundError:
        print("Error: final_metadata.csv not found")
        return
    
    # Step 2: Check if OSM graph already exists
    osm_graph_path = Path('component2/osm_road_network.graphml')
    
    if osm_graph_path.exists():
        print("\nStep 2: Loading existing OSM graph...")
        G = ox.load_graphml(osm_graph_path)
        print(f"✓ Loaded OSM graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("\nStep 2: Building OSM graph from scratch...")
        
        # Prepare nodes
        nodes = df.apply(lambda row: {
            'original_index': row['index'] if 'index' in row else row.name,
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'road_condition': row['road_condition'] if row['road_condition'] != 'none' else 'safe'
        }, axis=1).tolist()
        
        # Build OSM graph
        G = build_osm_road_network(nodes, network_type='drive')
        print(f"✓ Built OSM graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 3: Map images to edges
    edge_to_images = map_images_to_edges(G, df, max_distance=50)
    
    # Step 4: Load YOLO model
    model = load_yolo_model('component3/best.pt')
    
    # Step 5: Calculate safety scores
    edge_safety_scores = calculate_edge_safety_scores(G, edge_to_images, model, df)
    
    # Step 6: Save everything
    save_outputs(G, edge_to_images, edge_safety_scores)
    
    print("\n" + "="*60)
    print("✓ PROCESSING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • component4/data/osm_graph_base.graphml")
    print("  • component4/data/edge_to_images.json")
    print("  • component4/data/edge_safety_scores.json")
    print("  • component4/data/osm_graph_with_yolo.graphml")
    print("  • component4/data/osm_graph_with_yolo.gpickle")
    print("\nTo use in pathfinding:")
    print("  import osmnx as ox")
    print("  G = ox.load_graphml('component4/data/osm_graph_with_yolo.graphml')")
    print("  path = nx.shortest_path(G, source, target, weight='weight')")

if __name__ == "__main__":
    main()