import pandas as pd
import folium
from folium import plugins
import base64
import os
from pathlib import Path

class RoadSafetyMapper:
    def __init__(self, metadata_csv_path, images_folder='mapillary_images'):
        self.df = pd.read_csv(metadata_csv_path)
        self.images_folder = Path(images_folder)
        
        # Color scheme for road conditions
        self.colors = {
            'safe': 'green',
            'minor_issues': 'yellow',
            'major_problems': 'red'
        }
        
        # Calculate map center
        self.center_lat = self.df['latitude'].mean()
        self.center_lon = self.df['longitude'].mean()
    
    def sample_markers(self, safe_sample_rate=1.0):
        # Separate by road condition
        problems = self.df[self.df['road_condition'].isin(['minor_issues', 'major_problems'])]
        safe = self.df[self.df['road_condition'] == 'safe']
        
        # Sample safe points evenly
        if safe_sample_rate < 1.0:
            safe_sampled = safe.sample(n=int(len(safe) * safe_sample_rate), random_state=42)
        else:
            safe_sampled = safe
        
        # Combine: all problems + sampled safe
        sampled_df = pd.concat([problems, safe_sampled]).sort_values('index')
        
        return sampled_df
    
    def create_base_map(self, zoom_start=14):
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 60px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; padding: 10px">
        <b>Smart City Road Safety Monitoring System</b><br>
        St. Catharines, Ontario - Component 2: Image Mapping
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def add_legend(self, m):
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <b>Road Condition</b><br>
        <svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="green" stroke="green" stroke-width="2"/></svg> Safe<br>
        <svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="orange" stroke="orange" stroke-width="2"/></svg> Minor Issues<br>
        <svg width="20" height="20"><circle cx="10" cy="10" r="6" fill="red" stroke="red" stroke-width="2"/></svg> Major Problems
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_popup_content(self, row, embed_image=False):
        img_path = self.images_folder / row['filename']
        
        # Create image display
        if embed_image and img_path.exists():
            # Encode image as base64 for embedding (slow, large file)
            try:
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{img_data}" width="300px"><br>'
            except Exception as e:
                img_html = f'<i>Error loading image: {str(e)}</i><br>'
        elif img_path.exists():
            # Link to image file (fast, requires images in same directory structure)
            img_html = f'<a href="{img_path}" target="_blank">📷 View Image</a><br>'
        else:
            img_html = f'<i>Image not found: {row["filename"]}</i><br>'
        
        # Create metadata display
        popup_html = f'''
        <div style="width: 320px">
            {img_html}
            <b>Filename:</b> {row['filename']}<br>
            <b>Road Condition:</b> <span style="color:{self.colors.get(row['road_condition'], 'gray')}">{row['road_condition']}</span><br>
            <b>Problem Types:</b> {row['problem_types']}<br>
            <b>Direction:</b> {row['direction']}<br>
            <b>Coordinates:</b> ({row['latitude']:.5f}, {row['longitude']:.5f})<br>
            <b>Image Index:</b> {row['index']}
        </div>
        '''
        
        return popup_html
    
    def add_image_markers(self, m, data_df=None, add_popups=True, embed_images=True):
        if data_df is None:
            data_df = self.df
            
        # Create feature groups for each condition (allows layer control)
        safe_group = folium.FeatureGroup(name='Safe Roads')
        minor_group = folium.FeatureGroup(name='Minor Issues')
        major_group = folium.FeatureGroup(name='Major Problems')
        
        groups = {
            'safe': safe_group,
            'minor_issues': minor_group,
            'major_problems': major_group
        }
        
        print(f"Adding {len(data_df)} image markers...")
        
        for idx, row in data_df.iterrows():
            if pd.isna(row['road_condition']):
                continue
                
            color = self.colors.get(row['road_condition'], 'gray')
            
            # Create circle marker
            marker = folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2,
                tooltip=f"Image {row['index']}: {row['road_condition']}"
            )
            
            # Add popup if requested
            if add_popups:
                popup_html = self.create_popup_content(row, embed_image=embed_images)
                marker.add_child(folium.Popup(popup_html, max_width=350))
            
            # Add to appropriate group
            marker.add_to(groups.get(row['road_condition'], safe_group))
        
        # Add all groups to map
        for group in groups.values():
            group.add_to(m)
        
    def add_road_condition_overlay(self, m):        
        # Group by sequence_id to draw connected paths
        for sequence_id, group in self.df.groupby('sequence_id'):
            # Sort by index to maintain order
            group = group.sort_values('index')
            
            # Create line segments between consecutive points
            for i in range(len(group) - 1):
                row1 = group.iloc[i]
                row2 = group.iloc[i + 1]
                
                # Use the worse condition of the two points
                conditions = [row1['road_condition'], row2['road_condition']]
                if 'major_problems' in conditions:
                    color = self.colors['major_problems']
                elif 'minor_issues' in conditions:
                    color = self.colors['minor_issues']
                else:
                    color = self.colors['safe']
                
                # Draw line segment
                folium.PolyLine(
                    locations=[
                        [row1['latitude'], row1['longitude']],
                        [row2['latitude'], row2['longitude']]
                    ],
                    color=color,
                    weight=4,
                    opacity=0.7,
                    tooltip=f"Segment {i}: {row1['road_condition']} → {row2['road_condition']}"
                ).add_to(m)
            
    def create_full_map(self, output_file='road_safety_map.html', add_popups=True, 
                       embed_images=True, sample_safe_areas=True, safe_sample_rate=0.1):
        
        # Sample markers if requested
        if sample_safe_areas:
            marker_data = self.sample_markers(safe_sample_rate)
        else:
            marker_data = self.df
            print(f"\nUsing all {len(marker_data)} markers (no sampling)")
        
        # Create base map
        m = self.create_base_map()
        
        # Add road condition overlay first (so markers appear on top)
        # Use full dataset for overlay to show complete paths
        self.add_road_condition_overlay(m)
        
        # Add sampled markers
        self.add_image_markers(m, data_df=marker_data, add_popups=add_popups, embed_images=embed_images)
        
        # Add legend
        self.add_legend(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        # Save map
        m.save(output_file)
        
        return m


# ============================================================================
# HELPER FUNCTIONS FOR COMPONENT 4 INTEGRATION
# ============================================================================

def add_route_to_map(m, route_coords, color='blue', weight=5, label='Route'):
    folium.PolyLine(
        locations=route_coords,
        color=color,
        weight=weight,
        opacity=0.8,
        popup=label
    ).add_to(m)


def highlight_path_segment(m, start_coord, end_coord, color='purple', weight=6):
    folium.PolyLine(
        locations=[start_coord, end_coord],
        color=color,
        weight=weight,
        opacity=0.9,
        dash_array='10, 5'
    ).add_to(m)


if __name__ == "__main__":
    # Initialize mapper
    mapper = RoadSafetyMapper(
        metadata_csv_path='final_metadata.csv',
        images_folder='mapillary_images'
    )
    
    road_map = mapper.create_full_map(
        output_file='component2_road_safety_map.html',
        add_popups=True,
        embed_images=False,
        sample_safe_areas=True,  # Enable intelligent sampling
        safe_sample_rate=1.0    # Keep 10% of safe points (all problems kept)
    )