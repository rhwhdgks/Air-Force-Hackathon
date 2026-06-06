# Air Force Hackathon

강남구 생활권 데이터를 기반으로 쓰레기 발생 추정량을 시각화하고, 신규 쓰레기통 설치 후보지를 시뮬레이션하는 Flask 웹 애플리케이션입니다.

## 주요 기능

- 강남구 격자별 쓰레기 발생 추정량 지도 시각화
- 기존 쓰레기통, CCTV, 단속 데이터 기반 지도 오버레이
- 후보 지점 중 쓰레기통 설치 위치를 탐색하는 Greedy + swap 기반 시뮬레이션
- Server-Sent Events(SSE)를 이용한 진행률 및 결과 스트리밍

## 프로젝트 구조

```text
.
├── app.py                  # Flask 애플리케이션 진입점
├── data/gangnam/           # 원본 및 가공 CSV/GeoJSON 데이터
├── map/                    # 앱에서 직접 읽는 JSON/GeoJSON 지도 데이터
├── scripts/                # 데이터 가공, 후보지 생성, 시각화 스크립트
│   └── legacy/             # 이전 Flask 서버 버전
├── static/                 # HTML 지도, 이미지, 비디오 정적 파일
├── Project Overview.pdf    # 프로젝트 요약 자료
├── Project Proposal.pdf    # 프로젝트 제안서
└── requirements.txt        # Python 의존성
```

## 실행 방법

Python 3.9 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

서버가 실행되면 브라우저에서 `http://localhost:8080`으로 접속합니다.

## 주요 엔드포인트

- `/` : 최종 지도 화면
- `/gangnam` : 강남구 CCTV 포함 지도 화면
- `/api/markers` : 강남구 마커 GeoJSON 반환
- `/api/cctvs` : CCTV 위치 JSON 반환
- `/api/runGreedyStream?n=1000&r=50&max=1` : 쓰레기통 설치 시뮬레이션 실행
- `/api/overlayStream` : 시뮬레이션 결과 오버레이 SSE 스트림
- `/api/latestOverlay` : 최근 시뮬레이션 오버레이 결과

## 데이터

앱은 실행 시 `map/pred_50m_50cap.json`을 기본 격자 데이터로 읽습니다. 시뮬레이션 결과는 실행 중 `map/coverage_diff.json`으로 생성될 수 있으며, 이 파일은 재생성 가능한 결과물이므로 Git 추적 대상에서 제외했습니다.

## 정리 내용

- 저장소에 포함되어 있던 가상환경(`venv/`)과 OS/에디터 임시 파일을 제거했습니다.
- 최신 서버 파일을 `app.py`로 정리했습니다.
- 이전 서버 버전은 `scripts/legacy/`에 보관했습니다.
- 실행에 필요한 의존성을 `requirements.txt`에 보강했습니다.
