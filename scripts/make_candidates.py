#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12 만 포인트 → MiniBatch K-Means K=5 000
클러스터마다 waste_estimate 최대 1 개 + 원본 인덱스까지 저장
→ map/candidates_kmeans.json
"""
import os, pandas as pd
from sklearn.cluster import MiniBatchKMeans

ROOT     = os.path.dirname(os.path.abspath(__file__))           # scripts/
MAP_DIR  = os.path.join(ROOT, os.pardir, 'map')
SRC_JSON = os.path.join(MAP_DIR, 'GangNam_garbage_pred_10m.json')
OUT_JSON = os.path.join(MAP_DIR, 'candidates_kmeans_10000.json')

K      = 10000     # 후보 개수
BATCH  = 20000

# 1) 원본 로드  ─────────────────────────────────────────────
print("📥 loading", SRC_JSON)
df = pd.read_json(SRC_JSON, encoding='utf-8')        # 120 만 × 3
df = df.reset_index().rename(columns={'index':'origin_idx'})   # ★ 원본 인덱스 보존
print(f"→ {len(df):,} rows")

# 2) MiniBatch K-Means ────────────────────────────────────
print(f"🔄 MiniBatch K-Means  (K={K:,})")
coords = df[['lat','lng']].values
km     = MiniBatchKMeans(n_clusters=K, batch_size=BATCH, random_state=42).fit(coords)
df['cluster'] = km.labels_

# 3) 클러스터 대표점( waste 최댓값 ) 선택 ──────────────────
idx  = df.groupby('cluster')['waste_estimate'].idxmax()
cnd  = df.loc[idx, ['lat','lng','waste_estimate','origin_idx']]   # ★ origin_idx 포함
print("✅ candidates :", len(cnd))

# 4) 저장 ─────────────────────────────────────────────────
cnd.to_json(OUT_JSON, orient='records', force_ascii=False, indent=2)
print("💾 saved to", OUT_JSON)
