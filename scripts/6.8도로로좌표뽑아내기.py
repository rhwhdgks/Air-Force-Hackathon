import os
import osmnx as ox
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Point

# --- 설정 ---
STEP = 10  # 샘플링 간격 (m)

# 1. 강남구내 보행자 및 주요 도로망 추출
G = ox.graph_from_place(
    "Gangnam-gu, Seoul, South Korea",
    network_type='all'
)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# 2. 도로 유형 필터링 (보행자 및 일반 도로)
road_types = [
    'footway', 'pedestrian', 'path','living_street',
    'service', 'track',
    'residential', 'unclassified',
]
major_roads = edges[edges['highway'].apply(
    lambda v: any(rt in (v if isinstance(v, list) else [v]) for rt in road_types)
)]

# 3. 미터 CRS로 변환
major_roads_m = major_roads.to_crs(epsg=5179)

# 4. 1m 간격으로 포인트 샘플링
points = []
for geom in major_roads_m.geometry:
    lines = [geom] if geom.geom_type == 'LineString' else list(geom)
    for line in lines:
        length = line.length
        dists = np.arange(0, length, STEP)
        for d in dists:
            pt = line.interpolate(d)
            points.append(pt)

# 5. GeoDataFrame 생성 및 위경도로 변환
gdf_pts = gpd.GeoDataFrame(geometry=points, crs=major_roads_m.crs)
gdf_pts = gdf_pts.to_crs(epsg=4326)

df_out = gpd.pd.DataFrame({
    'lat': gdf_pts.geometry.y,
    'lng': gdf_pts.geometry.x
})
# 중복 제거
df_out = df_out.drop_duplicates(subset=['lat','lng']).reset_index(drop=True)

# 6. JSON 저장
output_json = os.path.join(os.getcwd(), 'sidewalk_points_10.json')
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(df_out.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

print(f"✅ 쓰레기통 후보 지점(sidewalk_points.json) 생성 완료: 총 {len(df_out)}개 포인트 (STEP={STEP}m)")
