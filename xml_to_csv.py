import xml.etree.ElementTree as ET
import csv

def parse_annotations_xml(xml_file, output_csv):
    """
    Parse CVAT annotations XML and create CSV with road_condition and problem_types.
    
    Args:
        xml_file: Path to the input XML file
        output_csv: Path to the output CSV file
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Problem types to look for
    problem_types = {'pothole', 'crack', 'flooding', 'construction', 'debris'}
    
    # Prepare data for CSV
    rows = []
    
    # Iterate through all image tags
    for image in root.findall('image'):
        image_name = image.get('name')
        road_condition = None
        problems = []
        
        # Get road_condition from tag
        for tag in image.findall('tag'):
            if tag.get('label') == 'road_condition':
                condition_attr = tag.find("attribute[@name='condition']")
                if condition_attr is not None:
                    road_condition = condition_attr.text
        
        # Get problem types from box labels
        for box in image.findall('box'):
            label = box.get('label')
            if label in problem_types:
                problems.append(label)
        
        # Create row with semicolon-separated problems
        problem_str = ';'.join(problems) if problems else ''
        
        rows.append({
            'image_name': image_name,
            'road_condition': road_condition or '',
            'problem_types': problem_str
        })
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'road_condition', 'problem_types']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully created {output_csv} with {len(rows)} rows")

if __name__ == "__main__":
    # Usage
    input_file = "annotations.xml"
    output_file = "road_annotations.csv"
    
    parse_annotations_xml(input_file, output_file)