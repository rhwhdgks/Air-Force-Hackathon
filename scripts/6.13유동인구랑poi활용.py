#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
격자별 foot-traffic + POI Gaussian kernel 로
point-level predicted_waste 산출
────────────────────────────────────────
• traffic 값 : point 가 속한 100×100 m 격자의 traffic 컬럼 사용
• POI        : 반경 R_POI 까지 σ(=SIGMA) 가우시안 가중 합
               w = exp(−d² / 2σ²)
"""

import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point

# ─── 0. 파일 경로 ─────────────────────────────────────
ROOT   = os.path.dirname(__file__)
MAPDIR = os.path.join(ROOT, 'map')

SIDE_JSON  = os.path.join(MAPDIR, 'sidewalk_points.json')        # lat·lng
GRID_GEOJS = os.path.join(MAPDIR, 'grid_with_traffic.geojson')   # 격자 + traffic
POI_CSV    = os.path.join(MAPDIR, 'GangNam POI Data.csv')        # lat·lng·category
OUT_JSON   = os.path.join(MAPDIR, 'sidewalk_waste_estimates_reg.json')

# ─── 1. 회귀계수 & 커널 파라미터 (★) ──────────────────
coeffs = {
    'traffic' : 0.018,     # 격자 traffic 1명당
    '카페'    : 0.6,
    '편의점'  : 0.45,
    '음식점'  : 0.35,
    '학교'    : 0.30,
    '공공기관' : 0.0,
    '관광명소' : 0.0,
    '대형마트' : 0.0,
    '문화시설' : 0.0,
    '병원' : 0.0,
    '숙박' : 0.0,
    '약국' : 0.0,
    '어린이집, 유치원' : 0.0,
    '은행' : 0.0,
    '주유소' : 0.0,
    '주차장' : 0.0,
    '중개업소' : 0.0,
    '지하철역' : 0.0,
    '학원' : 0.0
    # … 필요한 카테고리 추가
}
SIGMA   = 50.0   # (m)  |  가우시안 σ
R_POI   = 150.0  # (m)  |  커널 적용 최대 반경 (3σ 정도)

# ─── 2. 데이터 로드 ───────────────────────────────────
print("로드중 …")
df_side = pd.read_json(SIDE_JSON)                      # lat,lng
gdf_side = gpd.GeoDataFrame(
    df_side, geometry=gpd.points_from_xy(df_side.lng, df_side.lat), crs="EPSG:4326"
)

gdf_grid = gpd.read_file(GRID_GEOJS)                  # geometry + traffic
gdf_grid = gdf_grid.to_crs(epsg=4326)

df_poi = pd.read_csv(POI_CSV)                         # longitude, latitude, category
gdf_poi = gpd.GeoDataFrame(
    df_poi,
    geometry=gpd.points_from_xy(df_poi.longitude, df_poi.latitude),
    crs="EPSG:4326"
)

# ─── 3. 유동인구: point → 소속 격자 traffic 매핑 ───────
print("   • 격자 spatial join(traffic)…")
gdf_side = gpd.sjoin(gdf_side, gdf_grid[['traffic','geometry']],
                     how='left', predicate='within')
gdf_side['traffic'] = gdf_side['traffic'].fillna(0)

# ─── 4. POI Gaussian   (투영 → KDTree) ────────────────
print("   • POI Gaussian kernel…")
# metre CRS
gdf_side_m = gdf_side.to_crs(epsg=5179)
gdf_poi_m  = gdf_poi.to_crs(epsg=5179)

# 카테고리 리스트
cat_list = [c for c in coeffs if c != 'traffic']

# KD-Tree에 모든 POI 좌표(5179) 저장
poi_xyz = np.column_stack([gdf_poi_m.geometry.x, gdf_poi_m.geometry.y])
tree = cKDTree(poi_xyz)

for cat in cat_list:
    # 해당 카테고리 POI 인덱스
    idx_cat = np.where(gdf_poi_m['category'] == cat)[0]
    if len(idx_cat) == 0:
        gdf_side[cat] = 0.0
        continue

    # KDTree 서브뷰
    sub_tree = cKDTree(poi_xyz[idx_cat])

    # 각 보도 포인트 → 반경 R_POI 이내 거리들
    print(f"      · {cat} ({len(idx_cat):,}개)…")
    distances, _ = sub_tree.query(
        np.column_stack([gdf_side_m.geometry.x, gdf_side_m.geometry.y]),
        k=None, distance_upper_bound=R_POI
    )
    # distances 는 list of arrays; 빈 셀은 ∞ 포함 → 필터
    weighted = []
    for dist_arr in distances:
        if np.isinf(dist_arr).all():
            weighted.append(0.0)
        else:
            d = dist_arr[~np.isinf(dist_arr)]
            w = np.exp(- (d**2) / (2*SIGMA**2))
            weighted.append(w.sum())
    gdf_side[cat] = weighted

# ─── 5. waste_estimate 계산 ───────────────────────────
print("   • waste_estimate 계산…")
gdf_side['waste_estimate'] = (
    coeffs['traffic'] * gdf_side['traffic'] +
    sum(coeffs[c] * gdf_side[c] for c in cat_list)
)

# ─── 6. 저장 ─────────────────────────────────────────
out = gdf_side[['lat','lng','waste_estimate']].to_dict(orient='records')
with open(OUT_JSON,'w',encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"✅ {OUT_JSON}  ({len(out):,} points) 저장 완료")
