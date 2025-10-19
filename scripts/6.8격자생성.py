import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import linemerge, unary_union
import folium

# 1. 강남구 경계 및 secondary 이상 도로 추출
gangnam = ox.geocode_to_gdf("Gangnam-gu, Seoul, South Korea")
G = ox.graph_from_place("Gangnam-gu, Seoul, South Korea", network_type='walk')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
major_edges = edges[edges['highway'].isin(['motorway', 'trunk', 'primary', 'secondary'])]

# 2. 좌표계 설정 및 투영
grid_size = 100  # meters
projected_crs = "EPSG:5179"
gangnam_proj = gangnam.to_crs(projected_crs)
edges_proj = major_edges.to_crs(projected_crs)
boundary = gangnam_proj.geometry.iloc[0]
minx, miny, maxx, maxy = boundary.bounds

# 3. 도로 중심선으로 단순화
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree
import networkx as nx
import numpy as np

lines = list(edges_proj.geometry)
centers = [line.interpolate(0.5, normalized=True).coords[0] for line in lines]
tree = cKDTree(centers)

# 가까운 선끼리 네트워크 연결
pairs = tree.query_pairs(r=50)  # 15m 이내면 같은 도로로 간주
G = nx.Graph()
G.add_nodes_from(range(len(lines)))
G.add_edges_from(pairs)

# 클러스터 단위로 병합
merged_lines = []
for group in nx.connected_components(G):
    group_lines = [lines[i] for i in group]
    merged = unary_union(group_lines)
    if isinstance(merged, LineString):
        merged_lines.append(merged)
    elif isinstance(merged, MultiLineString):
        merged_lines.extend(merged.geoms)

# 결과 GeoDataFrame 생성
centerline_edges_proj = gpd.GeoDataFrame(geometry=merged_lines, crs=projected_crs)

# 4. 격자 생성 및 중심선 도로에 따른 분할
cols = np.arange(minx, maxx, grid_size)
rows = np.arange(miny, maxy, grid_size)
grid_polygons = []

for x in cols:
    for y in rows:
        cell = box(x, y, x + grid_size, y + grid_size)
        if not boundary.intersects(cell):
            continue
        cell = boundary.intersection(cell)
        intersecting_roads = centerline_edges_proj[centerline_edges_proj.intersects(cell)]
        if not intersecting_roads.empty:
            roads_union = intersecting_roads.unary_union
            result = cell.difference(roads_union.buffer(0.01))  # 도로는 거의 두께 없이 처리
            if isinstance(result, (Polygon, MultiPolygon)):
                grid_polygons.extend(result.geoms if isinstance(result, MultiPolygon) else [result])
        else:
            grid_polygons.append(cell)

# 5. GeoDataFrame 생성 및 WGS84 변환
grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=projected_crs).to_crs("EPSG:4326")
center = gangnam.geometry.iloc[0].centroid
m = folium.Map(location=[center.y, center.x], zoom_start=14)

# 6. 격자 시각화
for geom in grid_gdf.geometry:
    if geom.is_empty:
        continue
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        if not poly.is_empty and poly.exterior and len(poly.exterior.coords) > 0:
            folium.Polygon(
                locations=[(pt[1], pt[0]) for pt in poly.exterior.coords],
                color='blue', fill=False, weight=1
            ).add_to(m)
# 7. Folium 지도 저장 (선택)
m.save("gangnam_split_grid_map.html")
print("✅ 지도 HTML 저장 완료: gangnam_split_grid_map.html")

# 8. GeoJSON으로 격자 저장
output_path = "map/grid.geojson"  # 원하는 경로로 수정
grid_gdf["grid_id"] = range(len(grid_gdf))
grid_gdf.to_file(output_path, driver="GeoJSON", encoding="utf-8")
print(f"✅ 격자 GeoJSON 저장 완료: {output_path}")