# -*- coding: utf-8 -*-
"""
Gangnam sidewalk_points_10.json → waste_estimate 추가 (tqdm 진행 표시)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm          # ★ 진행 막대

# ──────────────────────────────────────────────────────────────
# 1. 경로 설정
DATA_DIR   = Path("/Users/kojonghan/쓰레기통 배치 분석")
POI_CSV    = DATA_DIR / "data/gangnam/필터링 poi 데이터.csv"
POINTS_JS  = DATA_DIR / "map/sidewalk_points_10.json"
OUT_JS     = DATA_DIR / "map/sidewalk_points_10_waste.json"

# ──────────────────────────────────────────────────────────────
# 2. 회귀 계수 & 가중치
COEF = {
    "식음료업종":  0.0147,
    "교육시설":    0.0082,
    "지하철역":    0.0044,
    "관광명소":   -0.1096,
    "문화시설":   -0.0091,
    "공공기관":   -0.0801,
    "약국":        0.0237,
}
MULTIPLIER = {"지하철역": 5.0, "관광명소": 3.0}
# POI 세부 → 상위 카테고리 변환
def classify(row):
    cat = row["category"]       # 예: '관광명소'   # 예: '카페', '지하철역' ...
    
    # ----- 식음료업종 가중합 -------------------------------------
    if cat == "카페":
        return ("식음료업종",  4/25)
    elif cat == "음식점" :
        return ("식음료업종",  1/25)
    elif cat == "편의점" :
        return ("식음료업종", 20/25)
        
    # ----- 교육시설 ---------------------------------------------
    if cat == "어린이집" or cat == "학교":
            return ("교육시설", 1.0)
    
    # 나머지는 세부 → 상위 1:1 매핑 가정
    return (cat, 1.0)
print("작동중")
# 지하철역·관광명소 배수
MULTIPLIER = {"지하철역": 5.0, "관광명소": 3.0}

## ──────────────────────────────────────────────────────────────
# 3. 데이터 불러오기
# 3-1 POI
poi = pd.read_csv(POI_CSV)
poi = poi.rename(columns={"lon": "x", "lat": "y"})  # 열 이름 맞추기
poi["geometry"] = [Point(xy) for xy in zip(poi.x, poi.y)]
poi[["supercat", "inner_w"]] = poi.apply(classify, axis=1, result_type="expand")
poi = poi.dropna(subset=["supercat"])

# 3-2 보행 포인트 (lat/lng 배열 JSON)
with open(POINTS_JS, "r", encoding="utf-8") as f:
    coords = json.load(f)  # [{'lat':..., 'lng':...}, ...]

pts = gpd.GeoDataFrame(
    coords,
    geometry=[Point(c["lng"], c["lat"]) for c in coords],
    crs="EPSG:4326"
)

# ──────────────────────────────────────────────────────────────
# 4. 좌표계 변환
CRS_KOREA = "EPSG:5179"
poi_gdf = gpd.GeoDataFrame(poi, geometry="geometry", crs="EPSG:4326").to_crs(CRS_KOREA)
pts_gdf = pts.to_crs(CRS_KOREA)

poi_coords = np.vstack([poi_gdf.geometry.x, poi_gdf.geometry.y]).T
pt_coords  = np.vstack([pts_gdf.geometry.x, pts_gdf.geometry.y]).T
tree = cKDTree(poi_coords)

# ──────────────────────────────────────────────────────────────
# 5. waste_estimate 계산 (tqdm 적용)
SIGMA  = 40.0   # m
RAD    = 300.0  # m
CONST  = 1.0 / (2 * SIGMA * SIGMA)

estimates = np.zeros(len(pts_gdf), dtype=float)
idx_lists = tree.query_ball_point(pt_coords, r=RAD)

for i, idxs in enumerate(tqdm(idx_lists, total=len(idx_lists), desc="waste_estimate")):
    if not idxs:
        continue
    sub_poi = poi_gdf.iloc[idxs]
    dxy = pt_coords[i] - poi_coords[idxs]
    dists = np.hypot(dxy[:, 0], dxy[:, 1])
    weights = np.exp(- (dists ** 2) * CONST)

    for w, (_, row) in zip(weights, sub_poi.iterrows()):
        cat   = row.supercat
        base  = COEF.get(cat, 0.0)
        inner = row.inner_w
        mult  = MULTIPLIER.get(cat, 1.0)
        estimates[i] += base * mult * inner * w

pts_gdf["waste_estimate"] = estimates

# ──────────────────────────────────────────────────────────────
# 6. plain JSON 저장 (lat/lng + waste_estimate)
pts_ll = pts_gdf.to_crs("EPSG:4326")
records = [
    {
        "lat": geom.y,
        "lng": geom.x,
        "waste_estimate": row.waste_estimate
    }
    for geom, row in zip(pts_ll.geometry, pts_ll.itertuples())
]

with open(OUT_JS, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ 완료! {OUT_JS} 저장")