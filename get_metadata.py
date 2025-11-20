import requests
import os
from time import sleep
from dotenv import load_dotenv
import csv

load_dotenv()

def get_sequence_metadata(sequence_key, access_token, csv_file="mapillary_data/combined_metadata.csv", start_index=0, interval=3):
    """
    Fetch sequence metadata and append to CSV file.
    
    Args:
        sequence_key: Mapillary sequence ID
        access_token: Your API token
        csv_file: Output CSV file (will append if exists)
        start_index: Starting index for this sequence (continues from previous)
        interval: Skip images to get one every N seconds (default: 3)
    """
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    url = f"https://graph.mapillary.com/image_ids"
    params = {
        "access_token": access_token,
        "sequence_id": sequence_key
    }
    
    response = requests.get(url, params=params)
    image_ids = response.json()["data"]
    print(f"Found {len(image_ids)} total images in sequence {sequence_key}")
    print(f"Sampling every {interval} images (keeping ~{len(image_ids)//interval} images)")
    
    metadata_list = []
    
    # Only process every Nth image
    for i, image_data in enumerate(image_ids):
        # Skip images that aren't at the interval
        if i % interval != 0:
            continue
            
        image_id = image_data.get('id') if isinstance(image_data, dict) else image_data
        
        try:
            image_url = f"https://graph.mapillary.com/{image_id}"
            params = {
                "access_token": access_token,
                "fields": "id,geometry,captured_at,compass_angle,altitude,width,height"
            }
            
            response = requests.get(image_url, params=params)
            data = response.json()
            
            coords = data.get("geometry", {}).get("coordinates", [None, None])
            
            # Use continuous index across all sequences
            continuous_index = start_index + i
            filename = f"{continuous_index:04d}_{image_id}.jpg"
            
            metadata = {
                "filename": filename,
                "image_id": image_id,
                "index": continuous_index,
                "latitude": coords[1],
                "longitude": coords[0],
                "direction": data.get("compass_angle"),
                "timestamp": data.get("captured_at"),
                "road_condition": "",
                "problem_type": "",
                "sequence_id": sequence_key  # Track which sequence this came from
            }
            
            metadata_list.append(metadata)
            print(f"Processed {len(metadata_list)} selected images (original index: {i})")
            
            sleep(0.05)
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue
    
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(csv_file)
    
    fieldnames = [
        "filename", "image_id", "index",
        "latitude", "longitude", "direction", "timestamp",
        "road_condition", "problem_type", "sequence_id"
    ]
    
    # Append to CSV
    with open(csv_file, "a", newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Only write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write rows with Excel-friendly formatting
        for row in metadata_list:
            row_copy = row.copy()
            # Format both image_id and timestamp as text for Excel
            row_copy['image_id'] = f'="{row["image_id"]}"'
            row_copy['timestamp'] = f'="{row["timestamp"]}"'
            writer.writerow(row_copy)
    
    print(f"Appended {len(metadata_list)} rows to {csv_file}")
    return len(metadata_list)


# Usage: Run multiple sequences one after another
if __name__ == "__main__":
    access_token = os.getenv("ACCESS_TOKEN")
    
    # List all your sequence keys in order
    sequence_keys = [
        
    ]
    
    current_index = 0
    csv_file = "mapillary_data/combined_metadata.csv"
    
    # Set your desired interval (3-5 seconds between images)
    # If Mapillary captures ~1 image per second, use:
    # interval = 3  # for 3 second gaps
    # interval = 4  # for 4 second gaps
    # interval = 5  # for 5 second gaps
    interval = 3  # Change this value as needed
    
    # Delete old combined file if starting fresh
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Removed existing {csv_file}\n")
    
    for seq_key in sequence_keys:
        if seq_key:  # Skip if env var not set
            print(f"\n{'='*60}")
            print(f"Processing sequence: {seq_key}")
            print(f"{'='*60}")
            
            count = get_sequence_metadata(seq_key, access_token, csv_file, current_index, interval)
            current_index += count
    
    print(f"\n{'='*60}")
    print(f"✓ Combined metadata saved to: {csv_file}")
    print(f"✓ Total images: {current_index}")
    print(f"{'='*60}")