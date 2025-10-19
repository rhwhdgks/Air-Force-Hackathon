import pandas as pd

def filter_gangnam(input_path, output_path):
    # 엑셀 파일 읽기 (한글 컬럼명 지원)
    df = pd.read_excel(input_path, dtype=str)
    
    # '소재지도로명주소' 컬럼에 '강남구' 포함된 행만 필터링
    mask = df['관리기관명'].str.contains('강남구청', na=False)
    filtered = df[mask]
    
    # 결과 확인 (원하면 첫 5행 출력)
    print(f"전체 행 개수: {len(df)}, 필터링된 행 개수: {len(filtered)}")
    print(filtered.head())
    
    # 엑셀로 저장
    filtered.to_excel(output_path, index=False)
    print(f"✅ '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # 본인 로컬 경로로 수정하세요
    input_path = "/Users/kojonghan/쓰레기통 배치 분석/data/gangnam/서울시 CCTV 현황.xlsx"
    output_path = "/Users/kojonghan/쓰레기통 배치 분석/data/gangnam/서울시_CCTV_강남구.xlsx"
    
    filter_gangnam(input_path, output_path)
