import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box

# --- 설정 ---
SIGMA = 100         # Gaussian 커널 시그마 (감쇠 속도, m)
RADIUS = 100        # 사각형 크기 반경 (m)
GRID_CRS = 5179    # 미터 CRS

# --- 유틸 함수 ---
def gaussian_kernel(distance, sigma=SIGMA):
    """
    거리(distance)에 대해 Gaussian 가중치를 계산
    """
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

# --- 데이터 로드 ---
def load_grid(grid_geojson, model_json_path):
    gdf = gpd.read_file(grid_geojson).to_crs(epsg=GRID_CRS)
    df_model = pd.read_json(model_json_path, encoding='utf-8')
    gdf['predicted_waste'] = df_model['predicted_waste'].values
    gdf['centroid'] = gdf.geometry.centroid
    return gdf


def load_sidewalk_points(points_json):
    df = pd.read_json(points_json, encoding='utf-8')
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lng, df.lat),
        crs='EPSG:4326'
    ).to_crs(epsg=GRID_CRS)
    return gdf

# --- 계산 로직 ---
def compute_estimates(gdf_grid, gdf_pts, radius=RADIUS):
    estimates = []
    total = len(gdf_pts)
    print(f"▶ 총 {total}개 포인트 계산 시작 (Gaussian σ={SIGMA}, 반경={radius}m)")
    for idx, pt in enumerate(gdf_pts.geometry):
        if idx % max(1, total // 10) == 0:
            print(f"  처리 중... {idx}/{total} ({idx*100//total}% 완료)")
        area = box(pt.x - radius, pt.y - radius, pt.x + radius, pt.y + radius)
        subset = gdf_grid[gdf_grid.geometry.intersects(area)]
        if subset.empty:
            estimates.append(0)
            continue
        distances = subset.centroid.distance(pt)
        weights = gaussian_kernel(distances)
        estimates.append((subset['predicted_waste'].values * weights).sum())
    return estimates

# --- 결과 저장 ---
def save_results(gdf_pts, estimates, output_json):
    gdf_pts['waste_estimate'] = estimates
    gdf_latlon = gdf_pts.to_crs(epsg=4326)
    df = pd.DataFrame({
        'lat': gdf_latlon.geometry.y,
        'lng': gdf_latlon.geometry.x,
        'waste_estimate': gdf_latlon['waste_estimate']
    })
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    print(f"✅ 결과 저장 완료: {output_json} (총 {len(df)}개 레코드)")

# --- 메인 실행 ---
if __name__ == '__main__':
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    grid_geojson = os.path.join(base, 'map', 'grid.geojson')
    model_json    = os.path.join(base, 'map', 'grid_model.json')
    points_json   = os.path.join(base, 'map', 'sidewalk_points.json')
    out_json      = os.path.join(base, 'map', 'sidewalk_waste_estimates_gaussian_100.json')

    # 로드
    gdf_grid = load_grid(grid_geojson, model_json)
    gdf_pts  = load_sidewalk_points(points_json)

    # 계산
    estimates = compute_estimates(gdf_grid, gdf_pts)

    # 1번 정규화: 총량 보존 스케일링
    total_model = gdf_grid['predicted_waste'].sum()
    sum_est = sum(estimates)
    if sum_est > 0:
        factor = total_model / sum_est
        estimates = [e * factor for e in estimates]
        print(f"▶ 총량 보존 정규화 적용: factor={factor:.6f}, 정규화 후 합계={sum(estimates):.2f}")
    else:
        print("▶ 정규화 생략: 계산된 합계가 0입니다.")

    # 저장
    save_results(gdf_pts, estimates, out_json)
