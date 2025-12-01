import requests
import os
import csv
from time import sleep
from dotenv import load_dotenv

load_dotenv()

def download_images_from_metadata(metadata_csv="combined_metadata.csv", 
                                   access_token=None,
                                   output_dir="mapillary_images", 
                                   use_original=False):
    """
    Download images using the combined metadata CSV with continuous numbering.
    This ensures filenames match the metadata exactly.
    """
    
    # os.makedirs(output_dir, exist_ok=True)
    
    # Read the combined metadata
    with open(metadata_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        metadata_rows = list(reader)
    
    total_images = len(metadata_rows)
    print(f"Found {total_images} images in metadata CSV")
    print(f"{'='*60}\n")
    
    for row_num, row in enumerate(metadata_rows, 1):
        # Extract image_id (remove Excel formatting if present)
        image_id = row['image_id'].strip().replace('="', '').replace('"', '')
        index = int(row['index'])
        filename = row['filename']
        
        try:
            # Get image metadata from Mapillary API
            image_url = f"https://graph.mapillary.com/{image_id}"
            
            if use_original:
                fields = "width,height,thumb_original_url,thumb_2048_url,thumb_1024_url,thumb_256_url"
            else:
                fields = "thumb_2048_url,thumb_1024_url,thumb_256_url"
            
            params = {
                "access_token": access_token,
                "fields": fields
            }
            
            response = requests.get(image_url, params=params)
            data = response.json()
            
            # Get the appropriate download URL
            image_download_url = None
            
            # Try original first, then fallback to thumbnails
            for field in ['thumb_original_url', 'thumb_2048_url', 'thumb_1024_url', 'thumb_256_url']:
                if field in data:
                    image_download_url = data[field]
                    res_info = f"{data.get('width')}x{data.get('height')}" if 'width' in data else field
                    break
            
            if not image_download_url:
                print(f"⚠ No image URL available for {image_id}, skipping")
                continue
            
            # Download image
            image_response = requests.get(image_download_url)
            image_response.raise_for_status()
            
            # Save with exact filename from metadata
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_response.content)
            
            if row_num % 10 == 0:
                print(f"Downloaded {row_num}/{total_images}: {filename}")
            
            # Be nice to the API
            sleep(0.1)
            
        except Exception as e:
            print(f"✗ Error downloading {image_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Download complete!")
    print(f"✓ Images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    download_images_from_metadata(
        metadata_csv="combined_metadata.csv",
        access_token=os.getenv("ACCESS_TOKEN"),
        use_original=False  # Set to True for original resolution
    )