import json
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from folium.plugins import MarkerCluster
import branca.colormap as cm

# 1) 데이터 로드
MODEL_JSON = "/Users/kojonghan/쓰레기통 배치 분석/map/sidewalk_waste_estimates_gaussian_100.json"  # 실제 경로로 조정
df = pd.read_json(MODEL_JSON, encoding="utf-8")

# 2) GeoDataFrame 생성
gdf = gpd.GeoDataFrame(
    df,
    geometry=df.apply(lambda row: Point(row["lng"], row["lat"]), axis=1),
    crs="EPSG:4326"
)

# 3) 컬러맵 준비 (그린→옐로→레드)
vmin = gdf["waste_estimate"].min()
vmax = gdf["waste_estimate"].max()
colormap = cm.LinearColormap(
    ["green", "yellow", "red"],
    vmin=vmin, vmax=vmax,
    caption="waste_estimate"
)

# 4) 지도 초기화
m = folium.Map(location=[37.5172, 127.0473], zoom_start=13)
colormap.add_to(m)

# 5) MarkerCluster 생성
marker_cluster = MarkerCluster(name="Grid Centers").add_to(m)

# 6) 클러스터 안에 CircleMarker 추가
for _, row in gdf.iterrows():
    waste = row["waste_estimate"]
    color = colormap(waste)

    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>waste_estimate :</b> {waste}",
            max_width=200
        )
    ).add_to(marker_cluster)

# 7) 레이어 컨트롤 및 저장
folium.LayerControl().add_to(m)
m.save("side_walk_clustered.html")
print("✅ grid_model_clustered.html 생성 완료")
