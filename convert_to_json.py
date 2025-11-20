import pandas as pd
import json

df = pd.read_csv('mapillary_data/combined_metadata.csv')

tasks = []
for idx, row in df.iterrows():
    task = {
        "data": {
            "image": f"http://localhost:8081/{row['filename']}",
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "timestamp": row['timestamp'],
            "direction": row['direction']
        }
    }
    tasks.append(task)

with open('label_studio_import.json', 'w') as f:
    json.dump(tasks, f, indent=2)

print(f"Created import file with {len(tasks)} tasks")
print("Next steps:")
print("1. Run the CORS server from mapillary_images directory")
print("2. Start Label Studio")
print("3. Import label_studio_import.json")