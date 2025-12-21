# COSC 3P71 Industry Project (Group #4 Submission)

## Group Members

- Owen Downing
- Tristan Froese
- Daniel Maspoch
- Nicholas Maspoch

## Overview

A working prototype of an intelligent road safety monitoring system that
demonstrates:

- Automated detection of road problems using computer vision
- Intelligent route planning based on safety considerations
- Real-time adaptation to changing road conditions
- Integration of multiple AI technologies in a practical application

## Setup

- Clone the repository
- Install dependencies

```
$ pip install -r requirements.txt
```

- Refer to Deliverables and Methodologies sections for required submission files and information

## Deliverables

### 1. Data Collection and Annotation

- [1,530 images](/dataset/images/all/)
- [CSV file with metadata and labels](/final_metadata.csv)

### 2. Geospatial Mapping and Graph Construction

- [Interactive map visualization](/component2_road_safety_map.html)
  - [Implementation](/component2/road_safety_mapper.py)
- [Graph data structure](/component2/graph.py)

### 3. AI Model Development and Fine-Tuning

- YOLOv8 model was used
- The following are all found in the [component3](/component3/) folder:
  - Trained model files
  - [Training](/component3/TRAIN.py) and [evaluation](/component3/EVALUATION.py) scripts
  - [Performance metrics and analysis report](/component3/YOLOv8_Performance_Analysis_Report.pdf)
  - [Model inference pipeline](/component3/INFERENCE.py)
  - [Results of runs](/runs/)

### 4. Intelligent Navigation System

- [Navigation system implementation](/component4/navigation_app.py)
- Navigation system can be accessed locally or online:

  - For local, run the following command in the root folder:

    ```
    $ streamlit run component4/navigation_app.py
    ```

- Streamlit Community Cloud link:
  https://cosc-3p71-industry-project-v4qgyreaug4e9vzrdn5xkg.streamlit.app/

## Methodologies

### Data Collection

- Mapillary for capturing geotagged photos
- CVAT (Computer Vision Annotation Tool) for data annotation

### Geospatial Mapping

- Folium for image mapping
- NetworkX for graph construction

### Navigation System

- Streamlit for user interface
