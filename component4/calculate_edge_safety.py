import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import sys
from pathlib import Path
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import graph building functions
from component2.graph import build_weighted_road_network_graph, update_edge_safety_from_yolo, normalize_road_conditions

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
    # Convert to radians for calculation
    px, py = point_lon, point_lat
    ax, ay = line_start_lon, line_start_lat
    bx, by = line_end_lon, line_end_lat
    
    # Vector from A to B
    AB_x = bx - ax
    AB_y = by - ay
    
    # Vector from A to Point
    AP_x = px - ax
    AP_y = py - ay
    
    # Calculate dot products
    AB_AB = AB_x * AB_x + AB_y * AB_y
    AP_AB = AP_x * AB_x + AP_y * AB_y
    
    if AB_AB == 0:
        # Line segment is actually a point
        return calculate_haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)
    
    # Calculate parameter t (projection of point onto line)
    t = max(0, min(1, AP_AB / AB_AB))
    
    # Find closest point on line segment
    closest_x = ax + t * AB_x
    closest_y = ay + t * AB_y
    
    # Calculate distance from point to closest point
    return calculate_haversine_distance(point_lat, point_lon, closest_y, closest_x)

def map_images_to_edges(G, images_df):
    """
    Map each image to its nearest graph edge.
    Returns: dict mapping (u, v) edge tuples to lists of image indices
    """
    print("\n" + "="*60)
    print("MAPPING IMAGES TO GRAPH EDGES")
    print("="*60)
    
    edge_to_images = {}
    edges = list(G.edges())
    
    # Initialize all edges with empty lists
    for edge in edges:
        u, v = edge
        if u < v:  # Only store one direction to avoid duplicates
            edge_to_images[(u, v)] = []
    
    print(f"\nProcessing {len(images_df)} images...")
    
    for idx, row in images_df.iterrows():
        img_lat = row['latitude']
        img_lon = row['longitude']
        img_index = row['index'] if 'index' in row else idx
        
        min_distance = float('inf')
        nearest_edge = None
        
        # Find nearest edge
        for u, v in edges:
            if u >= v:  # Skip reverse edges
                continue
                
            # Get node positions
            u_lon, u_lat = G.nodes[u]['pos']
            v_lon, v_lat = G.nodes[v]['pos']
            
            # Calculate distance from image to this edge
            dist = point_to_line_segment_distance(
                img_lat, img_lon,
                u_lat, u_lon,
                v_lat, v_lon
            )
            
            if dist < min_distance:
                min_distance = dist
                nearest_edge = (u, v)
        
        # Assign image to nearest edge
        if nearest_edge and min_distance < 100:  # Within 100m threshold
            edge_to_images[nearest_edge].append(img_index)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(images_df)} images...")
    
    # Print statistics
    total_mapped = sum(len(imgs) for imgs in edge_to_images.values())
    edges_with_images = sum(1 for imgs in edge_to_images.values() if len(imgs) > 0)
    
    print(f"\n✓ Mapping complete!")
    print(f"  • Total images mapped: {total_mapped}/{len(images_df)}")
    print(f"  • Edges with images: {edges_with_images}/{len(edge_to_images)}")
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
    
    Based on road_condition classes: safe, minor_issues, major_issues, major_problems
    """
    try:
        results = model(image_path, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get highest confidence detection
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Find the detection with highest confidence
            max_idx = np.argmax(confidences)
            max_conf = float(confidences[max_idx])
            predicted_class = int(classes[max_idx])
            
            # Map class to probability based on severity
            # Typical mapping: 0=safe, 1=minor_issues, 2=major_issues, 3=major_problems
            class_to_probability = {
                0: 0.1,   # safe
                1: 0.5,   # minor_issues
                2: 0.8,   # major_issues
                3: 0.9    # major_problems
            }
            
            base_prob = class_to_probability.get(predicted_class, 0.5)
            # Weight by confidence
            return base_prob * max_conf
        else:
            # No detections = safe road
            return 0.1
            
    except Exception as e:
        print(f"  Warning: Error processing {image_path}: {e}")
        return 0.2  # Default to relatively safe if error

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
            # No images for this edge, use default safe score
            edge_safety_scores[edge_tuple] = 0.2
            continue
        
        probabilities = []
        
        for img_idx in image_indices:
            # Find image by index in dataframe
            img_row = images_df[images_df['index'] == img_idx]
            
            if len(img_row) == 0:
                continue
            
            # Get filename directly from metadata
            img_filename = img_row.iloc[0]['filename']
            img_path = Path('dataset/images/all') / img_filename
            
            if img_path.exists():
                prob = run_yolo_inference_single(model, str(img_path))
                probabilities.append(prob)
            else:
                print(f"  Warning: Image not found: {img_path}")
        
        if probabilities:
            # Aggregate scores - using mean, but could use max or weighted average
            avg_prob = sum(probabilities) / len(probabilities)
            edge_safety_scores[edge_tuple] = avg_prob
            
            if processed % 10 == 0:
                print(f"  [{processed}/{total_edges}] Edge {edge_tuple}: "
                      f"{len(probabilities)} images, prob={avg_prob:.3f}")
        else:
            edge_safety_scores[edge_tuple] = 0.2
    
    print(f"\n✓ Calculated safety scores for {len(edge_safety_scores)} edges")
    
    # Print distribution
    scores = list(edge_safety_scores.values())
    print(f"\nScore Distribution:")
    print(f"  • Min:  {min(scores):.3f}")
    print(f"  • Mean: {np.mean(scores):.3f}")
    print(f"  • Max:  {max(scores):.3f}")
    print(f"  • Safe (p<0.3):           {sum(1 for s in scores if s < 0.3)}")
    print(f"  • Possibly Hazardous:     {sum(1 for s in scores if 0.3 <= s < 0.7)}")
    print(f"  • Hazardous (p>=0.7):     {sum(1 for s in scores if s >= 0.7)}")
    
    return edge_safety_scores

def save_outputs(G, edge_to_images, edge_safety_scores):
    """Save all outputs to files."""
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    output_dir = Path('component4/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base graph using pickle
    graph_base_path = output_dir / 'graph_base.gpickle'
    with open(graph_base_path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"✓ Base graph saved to: {graph_base_path}")
    
    # Save edge-to-images mapping
    mapping_path = output_dir / 'edge_to_images.json'
    # Convert tuple keys to strings for JSON
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
    print("\nUpdating graph with YOLO-based weights...")
    update_edge_safety_from_yolo(G, edge_safety_scores)
    
    # Save enhanced graph using pickle
    graph_yolo_path = output_dir / 'graph_with_yolo.gpickle'
    with open(graph_yolo_path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"✓ YOLO-enhanced graph saved to: {graph_yolo_path}")

def main():
    print("="*60)
    print("COMPONENT 4: EDGE SAFETY CALCULATION WITH YOLO")
    print("="*60)
    
    # Step 1: Load metadata
    print("\nStep 1: Loading metadata...")
    try:
        df = pd.read_csv('final_metadata.csv')
        print(f"✓ Loaded {len(df)} image records")
    except FileNotFoundError:
        print("Error: final_metadata.csv not found")
        return
    
    # Step 2: Build base graph
    print("\nStep 2: Building base graph from metadata...")
    
    if 'index' in df.columns:
        df = df.sort_values('index').reset_index(drop=True)
    
    nodes = df.apply(lambda row: {
        'original_index': row['index'] if 'index' in row else row.name,
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'road_condition': row['road_condition']
    }, axis=1).tolist()
    
    nodes = normalize_road_conditions(nodes)
    G = build_weighted_road_network_graph(nodes, simplify=True, 
                                         angle_threshold=15,
                                         averaging_threshold=75)
    print(f"✓ Graph built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 3: Map images to edges
    edge_to_images = map_images_to_edges(G, df)
    
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
    print("  • component4/data/graph_base.gpickle")
    print("  • component4/data/edge_to_images.json")
    print("  • component4/data/edge_safety_scores.json")
    print("  • component4/data/graph_with_yolo.gpickle")

if __name__ == "__main__":
    main()