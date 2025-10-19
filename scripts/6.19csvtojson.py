import pandas as pd

df = pd.read_csv("/Users/kojonghan/쓰레기통 배치 분석/data/gangnam/pred_50m_50cap.csv", encoding="utf-8")   # 필요하면 euc-kr 등으로 변경
df.to_json("/Users/kojonghan/쓰레기통 배치 분석/map/pred_50m_50cap.json", orient="records", force_ascii=False, indent=2)
