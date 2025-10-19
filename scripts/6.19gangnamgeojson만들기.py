#!/usr/bin/env python3
# build_gangnam_geojson.py
# 두 CSV(쓰레기통, 단속 지점) → gangnam_markers.geojson

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# --- 사용자 설정 ------------------------------------------------------------
SRC_BIN  = Path("/Users/kojonghan/쓰레기통 배치 분석/data/gangnam/강남구 쓰레기통.csv")
SRC_ENF  = Path("/Users/kojonghan/쓰레기통 배치 분석/data/gangnam/강남구 불법투기 단속.csv")
DST_GEO  = SRC_BIN.parent / "gangnam_markers.geojson"

# 각 CSV마다 “lat, lng, name”으로 맞춰 주는 매핑 (필요 없으면 {} 그대로)
col_map = {
    SRC_BIN : {"위도": "lat", "경도": "lng"},
    SRC_ENF : {"위도": "lat", "경도": "lng"},
}
# ---------------------------------------------------------------------------

def load_csv(path, mtype):
    """CSV → DataFrame + type 열 추가"""
    df = pd.read_csv(path)
    # 열 이름 치환
    if col_map.get(path):
        df = df.rename(columns=col_map[path])
    # 필요한 열만 남기고 결측 행 제거
    df = df[['lat', 'lng']].dropna(subset=['lat', 'lng'])
    df['type'] = mtype        # 'bin' 또는 'enforcement'
    return df

def main():
    bins = load_csv(SRC_BIN, 'bin')
    enf  = load_csv(SRC_ENF, 'enforcement')

    # 두 DF 결합
    df = pd.concat([bins, enf], ignore_index=True)

    # GeoDataFrame 생성 (EPSG:4326 = WGS84)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.lng, df.lat)],
        crs="EPSG:4326"
    )

    # 필요 속성만 남길 수도 있음
    # gdf = gdf[['type', 'name', 'geometry']]

    gdf.to_file(DST_GEO, driver="GeoJSON")
    print("✅ GeoJSON saved to", DST_GEO)

if __name__ == "__main__":
    main()
