import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import osmnx as ox
import networkx as nx
from ultralytics import YOLO

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

def load_osm_graph(filepath='osm_road_network.graphml'):
    """Load the OSM graph."""
    print(f"\nLoading OSM graph from {filepath}...")
    G = ox.load_graphml(filepath)
    print(f"✓ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def map_images_to_osm_edges(G, images_df, max_distance=50):
    """
    Map each image to its nearest OSM graph edge.
    Returns: dict mapping (u, v, key) edge tuples to lists of image indices
    """
    print("\n" + "="*60)
    print("MAPPING IMAGES TO OSM GRAPH EDGES")
    print("="*60)
    
    edge_to_images = {}
    
    print(f"\nProcessing {len(images_df)} images...")
    
    for idx, row in images_df.iterrows():
        img_lat = row['latitude']
        img_lon = row['longitude']
        img_index = row['index'] if 'index' in row else idx
        
        min_distance = float('inf')
        nearest_edge = None
        
        # Find nearest edge (including key for MultiDiGraph)
        for u, v, key in G.edges(keys=True):
            # Get node positions from OSM graph
            u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
            v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
            
            # Calculate distance from image to this edge
            dist = point_to_line_segment_distance(
                img_lat, img_lon,
                u_lat, u_lon,
                v_lat, v_lon
            )
            
            if dist < min_distance:
                min_distance = dist
                nearest_edge = (u, v, key)
        
        # Assign image to nearest edge if within threshold
        if nearest_edge and min_distance < max_distance:
            if nearest_edge not in edge_to_images:
                edge_to_images[nearest_edge] = []
            edge_to_images[nearest_edge].append(img_index)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(images_df)} images...")
    
    # Print statistics
    total_mapped = sum(len(imgs) for imgs in edge_to_images.values())
    edges_with_images = len(edge_to_images)
    
    print(f"\n✓ Mapping complete!")
    print(f"  • Total images mapped: {total_mapped}/{len(images_df)}")
    print(f"  • Edges with images: {edges_with_images}/{G.number_of_edges()}")
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
    Returns safety penalty multiplier (1.0 = safe, higher = more hazardous).
    """
    try:
        results = model(image_path, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            max_idx = np.argmax(confidences)
            max_conf = float(confidences[max_idx])
            predicted_class = int(classes[max_idx])
            
            # Map YOLO class to safety penalty
            # 0=safe (1.0), 1=minor_issues (1.5), 2=major_issues (3.0), 3=major_problems (5.0)
            class_to_penalty = {
                0: 1.0,   # safe
                1: 1.5,   # minor_issues
                2: 3.0,   # major_issues
                3: 5.0    # major_problems
            }
            
            base_penalty = class_to_penalty.get(predicted_class, 1.0)
            # Weight by confidence
            return 1.0 + (base_penalty - 1.0) * max_conf
        else:
            return 1.0  # No detections = safe
            
    except Exception as e:
        print(f"  Warning: Error processing {image_path}: {e}")
        return 1.0

def calculate_edge_safety_scores(G, edge_to_images, model, images_df):
    """
    Run YOLO inference on all images and calculate safety penalty per edge.
    Returns dict mapping (u, v, key) -> penalty multiplier
    """
    print("\n" + "="*60)
    print("CALCULATING EDGE SAFETY SCORES WITH YOLO")
    print("="*60)
    
    edge_safety_penalties = {}
    total_edges = len(edge_to_images)
    processed = 0
    
    for edge_tuple, image_indices in edge_to_images.items():
        processed += 1
        
        if not image_indices:
            edge_safety_penalties[edge_tuple] = 1.0
            continue
        
        penalties = []
        
        for img_idx in image_indices:
            img_row = images_df[images_df['index'] == img_idx]
            
            if len(img_row) == 0:
                continue
            
            img_filename = img_row.iloc[0]['filename']
            img_path = Path('dataset/images/all') / img_filename
            
            if img_path.exists():
                penalty = run_yolo_inference_single(model, str(img_path))
                penalties.append(penalty)
            else:
                print(f"  Warning: Image not found: {img_path}")
        
        if penalties:
            # Use MAXIMUM penalty (worst case) for safety-critical routing
            max_penalty = max(penalties)
            edge_safety_penalties[edge_tuple] = max_penalty
            
            if processed % 10 == 0:
                print(f"  [{processed}/{total_edges}] Edge {edge_tuple[:2]}: "
                      f"{len(penalties)} images, penalty={max_penalty:.3f}")
        else:
            edge_safety_penalties[edge_tuple] = 1.0
    
    print(f"\n✓ Calculated safety penalties for {len(edge_safety_penalties)} edges")
    
    # Print distribution
    penalties = list(edge_safety_penalties.values())
    print(f"\nPenalty Distribution:")
    print(f"  • Min:  {min(penalties):.3f}")
    print(f"  • Mean: {np.mean(penalties):.3f}")
    print(f"  • Max:  {max(penalties):.3f}")
    print(f"  • Safe (p≤1.2):           {sum(1 for p in penalties if p <= 1.2)}")
    print(f"  • Minor Issues (1.2<p<2): {sum(1 for p in penalties if 1.2 < p < 2)}")
    print(f"  • Hazardous (p≥2):        {sum(1 for p in penalties if p >= 2)}")
    
    return edge_safety_penalties

def update_graph_with_yolo_scores(G, edge_safety_penalties):
    """
    Update OSM graph edges with YOLO-derived safety penalties.
    Correctly handles MultiDiGraph with edge keys.
    """
    print("\n" + "="*60)
    print("UPDATING GRAPH WITH YOLO SCORES")
    print("="*60)
    
    updated_count = 0
    
    for (u, v, key), penalty in edge_safety_penalties.items():
        if G.has_edge(u, v, key):
            # Get original length
            length = G[u][v][key].get('length', 100.0)
            
            # Update edge attributes
            G[u][v][key]['yolo_safety_penalty'] = penalty
            G[u][v][key]['safety_penalty'] = penalty  # Keep consistent naming
            
            # Combine with existing penalty if present
            existing_penalty = G[u][v][key].get('safety_penalty_manual', 1.0)
            combined_penalty = max(penalty, existing_penalty)  # Use worst case
            
            # Update weight: distance × combined_penalty
            G[u][v][key]['weight'] = length * combined_penalty
            G[u][v][key]['has_yolo_data'] = True
            
            updated_count += 1
    
    print(f"✓ Updated {updated_count} edges with YOLO scores")
    
    # Ensure all edges have consistent attributes
    for u, v, key in G.edges(keys=True):
        if 'has_yolo_data' not in G[u][v][key]:
            length = G[u][v][key].get('length', 100.0)
            G[u][v][key]['yolo_safety_penalty'] = 1.0
            G[u][v][key]['safety_penalty'] = 1.0
            G[u][v][key]['weight'] = length * 1.0
            G[u][v][key]['has_yolo_data'] = False
    
    return G

def save_graph(G, filename='osm_road_network_with_yolo.graphml'):
    """Save the updated graph."""
    print(f"\n✓ Saving graph to {filename}...")
    ox.save_graphml(G, filepath=filename)
    print(f"✓ Graph saved successfully")

def main():
    print("="*60)
    print("YOLO SAFETY SCORE INTEGRATION")
    print("="*60)
    
    # Step 1: Load metadata
    print("\nStep 1: Loading metadata...")
    try:
        df = pd.read_csv('final_metadata.csv')
        print(f"✓ Loaded {len(df)} image records")
    except FileNotFoundError:
        print("Error: final_metadata.csv not found")
        return
    
    # Step 2: Load OSM graph
    print("\nStep 2: Loading OSM graph...")
    G = load_osm_graph('osm_road_network.graphml')
    
    # Step 3: Map images to edges
    edge_to_images = map_images_to_osm_edges(G, df, max_distance=50)
    
    # Step 4: Load YOLO model
    model = load_yolo_model('component3/best.pt')
    
    # Step 5: Calculate safety scores
    edge_safety_penalties = calculate_edge_safety_scores(G, edge_to_images, model, df)
    
    # Step 6: Update graph with YOLO scores
    G = update_graph_with_yolo_scores(G, edge_safety_penalties)
    
    # Step 7: Save updated graph
    save_graph(G, 'osm_road_network_with_yolo.graphml')

if __name__ == "__main__":
    main()