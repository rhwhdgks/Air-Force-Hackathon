import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

# 설정
GRID_JSON     = "map/grid_model.json"               # grid_id, lat, lng, predicted_waste
SIDEWALK_JSON = "map/sidewalk_points.json"          # lat, lng
OUTPUT_JSON   = "map/sidewalk_waste_estimates_50.json"
RADIUS        = 100    # m
SIGMA         = 50     # Gaussian σ
GRID_CRS      = 5179   # meter 좌표계
PTS_CRS       = 4326   # WGS84

def gaussian_kernel(d, sigma=SIGMA):
    return np.exp(- (d**2) / (2 * sigma**2))

# 1) 격자 데이터 로드 & 좌표 추출
df_grid = pd.read_json(GRID_JSON)
# GeoDataFrame 안 써도, centroid coords 배열만 추출합니다:
# (lon, lat) → meter CRS로 변환 후 x, y arrays
gdf_grid = gpd.GeoDataFrame(
    df_grid,
    geometry=df_grid.apply(lambda r: Point(r["lng"], r["lat"]), axis=1),
    crs=f"EPSG:{PTS_CRS}"
).to_crs(epsg=GRID_CRS)
grid_pts = np.vstack((
    gdf_grid.geometry.x.values,
    gdf_grid.geometry.y.values
)).T
wastes = df_grid["predicted_waste"].values

# 2) 보도 후보점 로드 & KD-Tree 생성
df_pts = pd.read_json(SIDEWALK_JSON)
gdf_pts = gpd.GeoDataFrame(
    df_pts,
    geometry=df_pts.apply(lambda r: Point(r["lng"], r["lat"]), axis=1),
    crs=f"EPSG:{PTS_CRS}"
).to_crs(epsg=GRID_CRS)
pt_coords = np.vstack((
    gdf_pts.geometry.x.values,
    gdf_pts.geometry.y.values
)).T
tree = cKDTree(pt_coords)

# 3) 격자별로 주변 포인트 찾아 정규화 분배
estimates = np.zeros(len(pt_coords), dtype=float)

for i, (gx, gy) in enumerate(grid_pts):
    # 3.1) 이 격자 반경 R 내 포인트 인덱스만
    idxs = tree.query_ball_point((gx, gy), r=RADIUS)
    if not idxs:
        continue
    # 3.2) 거리 계산
    pts_xy = pt_coords[idxs]
    dists = np.sqrt((pts_xy[:,0] - gx)**2 + (pts_xy[:,1] - gy)**2)
    # 3.3) Gaussian weight
    w = gaussian_kernel(dists, sigma=SIGMA)
    if w.sum() == 0:
        continue
    # 3.4) 정규화
    w_norm = w / w.sum()
    # 3.5) 해당 포인트들에 predicted_waste[i]만큼 분배
    estimates[idxs] += wastes[i] * w_norm

# 4) 결과를 다시 WGS84로 변환하여 저장
# (lat/lng 순서 유지)
out_df = pd.DataFrame({
    "lat": df_pts["lat"],
    "lng": df_pts["lng"],
    "waste_estimate": estimates
})
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print(f"✅ {OUTPUT_JSON} 생성 완료: 총 {len(out_df)}개 포인트")
