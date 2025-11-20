import xml.etree.ElementTree as ET
import csv

def parse_cvat_xml_to_csv(xml_file, output_csv):
    """
    Parse CVAT XML and create a CSV file with metadata and labels
    for the Smart City Road Safety project.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Prepare data for CSV
    rows = []
    
    # Define the problem types we're looking for
    problem_types = ['pothole', 'crack', 'flooding', 'construction', 'debris']
    
    for image in root.findall('image'):
        image_name = image.get('name')
        
        # Initialize row data
        row = {
            'filename': image_name,
            'image_id': '',
            'latitude': '',
            'longitude': '',
            'direction': '',
            'timestamp': '',
            'road_condition': '',
            'problem_types': []
        }
        
        # Extract metadata from the metadata point annotation
        for points in image.findall('points'):
            if points.get('label') == 'metadata':
                # Extract attributes from metadata point
                for attr in points.findall('attribute'):
                    attr_name = attr.get('name')
                    attr_value = attr.text if attr.text else ''
                    
                    if attr_name in row:
                        row[attr_name] = attr_value
        
        # Extract road_condition from tag
        for tag in image.findall('tag'):
            if tag.get('label') == 'road_condition':
                for attr in tag.findall('attribute'):
                    if attr.get('name') == 'condition':
                        row['road_condition'] = attr.text if attr.text else ''
        
        # Extract problem types from bounding boxes
        detected_problems = set()
        for box in image.findall('box'):
            label = box.get('label')
            if label in problem_types:
                detected_problems.add(label)
        
        # Also check polygons in case any were used
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            if label in problem_types:
                detected_problems.add(label)
        
        # Convert problem types set to comma-separated string
        row['problem_types'] = ','.join(sorted(detected_problems)) if detected_problems else ''
        
        rows.append(row)
    
    # Write to CSV
    fieldnames = ['filename', 'image_id', 'latitude', 'longitude', 
                  'direction', 'timestamp', 'road_condition', 'problem_types']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            # Write only the fields we want (excluding problem_types list)
            csv_row = {k: row[k] for k in fieldnames if k != 'problem_types'}
            csv_row['problem_types'] = row['problem_types']
            writer.writerow(csv_row)
    
    print(f"Successfully created {output_csv}")
    print(f"Total images: {len(rows)}")
    print(f"Images with road conditions: {sum(1 for r in rows if r['road_condition'])}")
    print(f"Images with problems: {sum(1 for r in rows if r['problem_types'])}")

# Usage
if __name__ == "__main__":
    parse_cvat_xml_to_csv('annotations.xml', 'project_dataset.csv')