import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster, HeatMap
import os

# Load fire data
df = pd.read_csv("data/portugal_fires.csv")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

# Center of the map
center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=6)

# Color mapping
confidence_colors = {
    "low": "yellow",
    "nominal": "orange",
    "high": "red"
}

# Add clustered markers with popups
marker_cluster = MarkerCluster(name="Fire Points (clustered)").add_to(m)

for _, row in gdf.iterrows():
    confidence = str(row.get("confidence", "unknown")).lower()
    color = confidence_colors.get(confidence, "gray")

    popup_text = f"""
    <b>Confidence:</b> {confidence.capitalize()}<br>
    <b>Brightness:</b> {row.get('brightness', 'n/a')}<br>
    <b>Date:</b> {row.get('acq_date', 'n/a')}<br>
    <b>Satellite:</b> {row.get('satellite', 'n/a')}
    """

    folium.CircleMarker(
        location=(row.geometry.y, row.geometry.x),
        radius=4,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(marker_cluster)

# Add heatmap layer
heatmap_points = gdf[["latitude", "longitude"]].dropna().values.tolist()
HeatMap(heatmap_points, name="Fire Heatmap", radius=15, blur=10).add_to(m)

# Add custom legend
legend_html = """
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 120px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;">
     <b>Confidence Level</b><br>
     🔴 High<br>
     🟠 Nominal<br>
     🟡 Low
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Add layer control toggle
folium.LayerControl().add_to(m)

# Save
os.makedirs("outputs", exist_ok=True)
output_file = "outputs/portugal_fires_map.html"
m.save(output_file)
print(f"✅ Map with legend, heatmap, and popups saved to {output_file}")
