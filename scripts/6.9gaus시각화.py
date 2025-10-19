import json
import folium
from folium.plugins import HeatMap

# 1) 데이터 로드
JSON_PATH = "/Users/kojonghan/쓰레기통 배치 분석/map/sidewalk_waste_estimates_gaussian_100.json"  # 실제 경로로 수정
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    points = json.load(f)

# 2) Folium 지도 초기화 (강남구 중심)
m = folium.Map(location=[37.5172, 127.0473], zoom_start=13)

# 3) HeatMap용 데이터 준비
# (A) 단순 밀집도
heat_data = [[pt['lat'], pt['lng']] for pt in points]

# (B) 예측 쓰레기량 가중치 적용하고 싶으면 아래 주석 해제
heat_data = [[pt['lat'], pt['lng'], pt.get('waste_estimate', 1)] for pt in points]

# 4) HeatMap 레이어 추가
HeatMap(
    heat_data,
    radius=8,       # 반경 (픽셀)
    blur=15,        # 블러 강도
    min_opacity=0.2,
    max_zoom=13
).add_to(m)

# 5) 결과를 HTML로 저장
OUT_HTML = "sidewalk_heatmap.html"
m.save(OUT_HTML)
print(f"✅ 히트맵이 {OUT_HTML} 로 저장되었습니다.")
