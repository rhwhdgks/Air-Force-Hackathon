import json
import os
import pandas as pd

# 1) 경로 설정
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = "/Users/kojonghan/쓰레기통 배치 분석/data/gangnam/서울시 강남구 안심이 CCTV 연계 현황.csv"   # 필요에 따라 경로 조정
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_PATH = os.path.join(OUTPUT_DIR, "cctvs.json")

# 2) CSV 로드
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# 3) 지도에 찍을 좌표만 추출
#    다른 정보(예: 설치 장소, 쓰레기 종류)가 필요하면 컬럼명을 추가하세요.
points = df[["lat", "lng"]]

# 4) JSON 레코드 형태로 변환
records = points.to_dict(orient="records")

# 5) JSON 파일로 저장
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ {JSON_PATH} 생성 완료: 총 {len(records)}개 포인트")
