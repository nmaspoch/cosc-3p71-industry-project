import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom

def csv_to_cvat_xml(csv_file, output_xml):
    root = ET.Element('annotations')
    
    # Add meta information
    meta = ET.SubElement(root, 'meta')
    job = ET.SubElement(meta, 'job')
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        print("Column headers found:", reader.fieldnames)
        
        for idx, row in enumerate(reader):
            image = ET.SubElement(root, 'image')
            image.set('id', str(idx))
            image.set('name', row['filename'])
            image.set('width', '1920')
            image.set('height', '1080')
            
            # Create a small point annotation to hold metadata
            # This will be visible in CVAT with all the metadata
            points = ET.SubElement(image, 'points')
            points.set('label', 'metadata')
            points.set('occluded', '0')
            points.set('points', '10.0,10.0')  # Small point in top-left corner
            points.set('z_order', '0')
            
            # Add metadata as attributes to the point
            metadata = {
                'image_id': row.get('image_id', ''),
                'latitude': row.get('latitude', ''),
                'longitude': row.get('longitude', ''),
                'direction': row.get('direction', ''),
                'timestamp': row.get('timestamp', ''),
                'sequence_id': row.get('sequence_id', ''),
                'road_condition': row.get('road_condition', ''),
                'problem_type': row.get('problem_type', '')
            }
            
            for name, value in metadata.items():
                value = value.strip()
                if value:
                    attr = ET.SubElement(points, 'attribute')
                    attr.set('name', name)
                    attr.text = value
    
    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"Successfully converted! Created {output_xml}")
    print(f"Processed {idx + 1} images")

# Usage
csv_to_cvat_xml('mapillary_data/combined_metadata.csv', 'cvat_annotations.xml')