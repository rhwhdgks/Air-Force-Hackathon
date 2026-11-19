# 무단투기 감지·쓰레기통 배치 최적화 플랫폼

제7회 공군 창의·혁신 아이디어 공모 해커톤 지정과제 "깨끗한 거리 조성을 위한 쓰레기 무단투기 감소 대책" 제안 프로젝트입니다 (2025.05 ~ 2025.08). CCTV 무단투기 감지, 공간정보 기반 발생량 예측, 쓰레기통 배치 최적화, 웹 시각화를 하나의 플랫폼으로 설계했습니다.

이 저장소에는 그중 강남구 공간분석 데이터, 쓰레기통 배치 시뮬레이터, 지도 시각화, 데이터 가공 스크립트가 들어 있습니다.

## 전체 설계

| 모듈 | 내용 | 이 저장소 |
|---|---|---|
| 무단투기 감지 | RTSP CCTV 영상 → YOLOv8(PyTorch) 탐지 → 좌표·이미지·시간 저장, 기준 신뢰도 이상이면 SENS API로 담당자 SMS | 미포함 |
| 데이터 통합 | 단속 이력, 서울시 50m 유동인구, 상업·비상업 POI, 도로, 기존 쓰레기통을 위치 공통키로 병합해 PostgreSQL 15/PostGIS에 적재 (Airflow ETL) | 가공 스크립트·결과 데이터만 포함 |
| 발생량 예측 | 분석 지점별 Gaussian kernel 거리 가중치 → Multi-Scale Regression으로 변수별 영향 반경 탐색 → Random Forest | 예측 결과 격자 데이터 포함 |
| 배치 최적화 | K-Means 후보지 선별 → Greedy 초기 배치 → 1·2-swap 지역 탐색, 설치 반경 내 중복 금지·지점당 최대 설치 수 제약 | 포함 |
| 웹 시각화 | Flask 서버, SSE로 진행률·결과 전송, Kakao Maps 지도 | 포함 |

제안서의 강남구 분석에서는 기존 쓰레기통 974개를 재배치하면 무단투기 단속 건수가 45.4% 줄어드는 것으로 추정했습니다.

## 시뮬레이터 기능

- 강남구 격자별 쓰레기 발생 추정량 로드 (`map/pred_50m_50cap.json`)
- 기존 쓰레기통, CCTV, 단속 지점 지도 레이어
- 쓰레기통 개수, 설치 반경, 지점당 최대 설치 수를 입력받아 Greedy + 1·2-swap 배치 계산
- SSE로 진행률과 결과를 스트리밍하고 지도에 오버레이

## 실행

Python 3.9 이상

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`http://localhost:8080`에서 확인합니다.

| 엔드포인트 | 내용 |
|---|---|
| `/` | 지도와 쓰레기통 배치 시뮬레이터 |
| `/gangnam` | 강남구 CCTV 포함 지도 |
| `/api/markers` | 강남구 마커 GeoJSON |
| `/api/cctvs` | CCTV 위치 JSON |
| `/api/runGreedyStream?n=1000&r=50&max=1` | 배치 시뮬레이션 실행 (SSE) |
| `/api/overlayStream` | 시뮬레이션 결과 오버레이 (SSE) |
| `/api/latestOverlay` | 최근 시뮬레이션 결과 |

시뮬레이션 중 생성되는 `map/coverage_diff.json`은 Git에서 제외했습니다.

## 구조

```text
app.py                  Flask 애플리케이션
data/gangnam/           원본·가공 CSV, GeoJSON
map/                    앱이 읽는 JSON/GeoJSON 지도 데이터
scripts/                데이터 가공, 후보지 생성, 시각화 (legacy/: 이전 서버 버전)
static/                 HTML 지도, 이미지, 영상
Project Overview.pdf    프로젝트 요약
Project Proposal.pdf    제안서
```
