import json
import os
import shutil
from pathlib import Path

def filter_images_by_interval(
    json_path,
    output_json_path,
    image_folder,
    output_folder,
    interval_seconds=3,
    copy_images=True
):
    """
    Filter Mapillary images by time interval and update JSON metadata.
    
    Args:
        json_path: Path to input metadata.json
        output_json_path: Path to save filtered metadata.json
        image_folder: Folder containing original images
        output_folder: Folder to copy filtered images to
        interval_seconds: Minimum time interval between kept images (default: 3)
        copy_images: Whether to copy image files (default: True)
    """
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Original dataset: {len(metadata)} images")
    
    # Filter images by time interval
    filtered_metadata = []
    last_timestamp = None
    
    for item in metadata:
        current_timestamp = item['timestamp']
        
        # Keep first image or if interval has passed
        if last_timestamp is None or (current_timestamp - last_timestamp) >= (interval_seconds * 1000):
            filtered_metadata.append(item)
            last_timestamp = current_timestamp
    
    print(f"Filtered dataset: {len(filtered_metadata)} images")
    print(f"Reduction: {len(metadata) - len(filtered_metadata)} images removed ({(1 - len(filtered_metadata)/len(metadata))*100:.1f}%)")
    
    # Save filtered metadata
    with open(output_json_path, 'w') as f:
        json.dump(filtered_metadata, f, indent=2)
    print(f"\nFiltered metadata saved to: {output_json_path}")
    
    # Copy filtered images if requested
    if copy_images:
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nCopying images to: {output_folder}")
        for i, item in enumerate(filtered_metadata):
            src_file = os.path.join(image_folder, item['filename'])
            dst_file = os.path.join(output_folder, item['filename'])
            
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: File not found - {src_file}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Copied {i + 1}/{len(filtered_metadata)} images...")
        
        print(f"✓ Copied {len(filtered_metadata)} images")
    
    # Print statistics
    print("\n--- Statistics ---")
    if len(filtered_metadata) > 1:
        time_diffs = []
        for i in range(1, len(filtered_metadata)):
            time_diff = (filtered_metadata[i]['timestamp'] - filtered_metadata[i-1]['timestamp']) / 1000
            time_diffs.append(time_diff)
        
        avg_interval = sum(time_diffs) / len(time_diffs)
        print(f"Average interval: {avg_interval:.1f} seconds")
        print(f"Min interval: {min(time_diffs):.1f} seconds")
        print(f"Max interval: {max(time_diffs):.1f} seconds")


def filter_by_nth_image(
    json_path,
    output_json_path,
    image_folder,
    output_folder,
    keep_every_n=3,
    copy_images=True
):
    """
    Alternative: Keep every Nth image (e.g., every 3rd image).
    
    Args:
        json_path: Path to input metadata.json
        output_json_path: Path to save filtered metadata.json
        image_folder: Folder containing original images
        output_folder: Folder to copy filtered images to
        keep_every_n: Keep every Nth image (default: 3)
        copy_images: Whether to copy image files (default: True)
    """
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Original dataset: {len(metadata)} images")
    
    # Keep every Nth image
    filtered_metadata = [metadata[i] for i in range(0, len(metadata), keep_every_n)]
    
    print(f"Filtered dataset: {len(filtered_metadata)} images (keeping every {keep_every_n}th image)")
    print(f"Reduction: {len(metadata) - len(filtered_metadata)} images removed ({(1 - len(filtered_metadata)/len(metadata))*100:.1f}%)")
    
    # Save filtered metadata
    with open(output_json_path, 'w') as f:
        json.dump(filtered_metadata, f, indent=2)
    print(f"\nFiltered metadata saved to: {output_json_path}")
    
    # Copy filtered images if requested
    if copy_images:
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nCopying images to: {output_folder}")
        for i, item in enumerate(filtered_metadata):
            src_file = os.path.join(image_folder, item['filename'])
            dst_file = os.path.join(output_folder, item['filename'])
            
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: File not found - {src_file}")
            
            if (i + 1) % 10 == 0:
                print(f"Copied {i + 1}/{len(filtered_metadata)} images...")
        
        print(f"✓ Copied {len(filtered_metadata)} images")


# Example usage
if __name__ == "__main__":
    # Method 1: Filter by time interval (recommended for varying speeds)
    filter_images_by_interval(
        json_path="mapillary_data/metadata.json",
        output_json_path="metadata_filtered_3s.json",
        image_folder="mapillary_images",
        output_folder="images_filtered_3s",
        interval_seconds=3,
        copy_images=True
    )
    
    # Method 2: Keep every Nth image (simpler, good for consistent speed)
    # Uncomment to use:
    """
    filter_by_nth_image(
        json_path="metadata.json",
        output_json_path="metadata_filtered_every3.json",
        image_folder="images",
        output_folder="images_filtered_every3",
        keep_every_n=3,
        copy_images=True
    )
    """