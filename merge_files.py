import xml.etree.ElementTree as ET
import pandas as pd

# Parse the XML file
tree = ET.parse('annotations.xml')
root = tree.getroot()

# Create dictionaries to store the data
road_conditions = {}
problem_types = {}

# Define problem types we care about
PROBLEM_LABELS = {'pothole', 'crack', 'flooding', 'construction', 'debris'}

# Parse each image in the XML
for image in root.findall('.//image'):
    index = int(image.get('id'))
    
    # Get road condition
    road_tag = image.find('.//tag[@label="road_condition"]')
    if road_tag is not None:
        condition_attr = road_tag.find('.//attribute[@name="condition"]')
        if condition_attr is not None:
            road_conditions[index] = condition_attr.text
    
    # Get problem types from box labels
    problems = []
    for box in image.findall('.//box'):
        label = box.get('label')
        if label in PROBLEM_LABELS:
            problems.append(label)
    
    # Store unique problems as comma-separated string
    if problems:
        problem_types[index] = ','.join(sorted(set(problems)))

# Read the CSV file
df = pd.read_csv('combined_metadata.csv')

# Add the road_condition column
df['road_condition'] = df['index'].map(road_conditions)

# Add the problem_types column
df['problem_types'] = df['index'].map(problem_types)

# Save the updated CSV
df.to_csv('final_metadata.csv', index=False)

print("CSV file updated successfully!")
print(f"\nTotal rows: {len(df)}")
print(f"Rows with road_condition: {df['road_condition'].notna().sum()}")
print(f"Rows with problem_types: {df['problem_types'].notna().sum()}")
print("\nSample of updated data:")
print(df[df['problem_types'].notna()][['index', 'filename', 'road_condition', 'problem_types']].head(10))