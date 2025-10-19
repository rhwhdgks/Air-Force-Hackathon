#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
강남구 쓰레기통 시뮬레이션 서버 (Flask + SSE)
  · Greedy → 1‑Swap → 2‑Swap
  · 실시간 진행률 / 결과 스트림
  · 시뮬 종료 시   coverage_diff.json  생성
    (-1 = 현실만, 0 = 동일, +1 = 시뮬만 커버)
2025‑06‑20 — coverageReady 플래그 추가, 코드 정리
"""

import os, json, random, time, itertools, queue
from math import radians, sin, cos, atan2, sqrt

import numpy as np
import pandas as pd
from flask import Flask, Response, request, send_from_directory, jsonify, stream_with_context
from scipy.spatial import cKDTree

# ───────────────────────── 경로 · 로드 ─────────────────────────
ROOT   = os.path.dirname(__file__)
MAPDIR = os.path.join(ROOT, 'map')
STATIC = os.path.join(ROOT, 'static')

ALL   = pd.read_json(os.path.join(MAPDIR, 'GangNam_garbage_pred_10m.json'))
CAND  = pd.read_json(os.path.join(MAPDIR, 'candidates_kmeans_10000.json'))
GRID  = ALL[['lat', 'lng']].copy()              # 10 m 격자 그대로 활용
GRID_LATLNG = GRID[['lat', 'lng']].values.astype(np.float32)
GRID_TREE   = cKDTree(GRID_LATLNG, leafsize=64)

LAT = ALL.lat.values.astype(np.float32)
LNG = ALL.lng.values.astype(np.float32)
W   = ALL.waste_estimate.values.astype(np.float32)
N_PT = len(ALL)
TOTAL_WASTE = int(np.sum(W))

cand_idx  = CAND.origin_idx.values.astype(np.int32)
cand_mask = np.zeros(N_PT, bool); cand_mask[cand_idx] = True

PT_TREE = cKDTree(np.c_[LAT, LNG], leafsize=64)

# ───────────────────────── 지오 헬퍼 ──────────────────────────

def haversine(lat1,lng1,lat2,lng2):
    """Great‑circle distance (m)"""
    R=6_371_000.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ, dλ = radians(lat2-lat1), radians(lng2-lng1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def r_lat(Rm):  return Rm/111_320.0

def r_lng(Rm, φ): return Rm/(111_320.0*cos(radians(φ)))

# ───────────────────────── Flask ────────────────────────────
app = Flask(__name__, static_folder=STATIC, static_url_path='')

# —— SSE overlay 큐 ——
overlay_q   = queue.Queue(maxsize=20)
last_overlay= None

def publish_overlay(payload: dict):
    global last_overlay
    last_overlay = payload
    msg = json.dumps(payload, ensure_ascii=False)
    try:
        overlay_q.put_nowait(msg)
    except queue.Full:
        overlay_q.get_nowait(); overlay_q.put_nowait(msg)

@app.route('/api/overlayStream')
def overlay_stream():
    def gen():
        while True:
            yield f"event: overlay\ndata: {overlay_q.get()}\n\n"
    return Response(stream_with_context(gen()), mimetype='text/event-stream')

@app.route('/api/latestOverlay')
def latest_overlay():
    return (jsonify(last_overlay) if last_overlay else (jsonify({'empty':True}),204))

# —— 정적 라우트 ——
@app.route('/')
def index(): return send_from_directory(STATIC, '6.15클러스터링개조버전.html')

@app.route('/map/<path:p>')
def maps(p): return send_from_directory(MAPDIR, p)

@app.route('/gangnam')
def gangnam(): return send_from_directory(STATIC, 'gangnam.html')

@app.route('/api/markers')
def api_markers():
    with open(os.path.join(MAPDIR,'gangnam_markers.geojson'),encoding='utf-8') as f:
        return jsonify(json.load(f))

# ───────────────────────── Coverage helper ──────────────────
COVER_R   = 100.0
COVER_LAT = r_lat(COVER_R)

def coverage_mask(points):
    """points: list[(lat,lng)] → bool[|GRID|]"""
    if not points:
        return np.zeros(len(GRID), bool)
    t = cKDTree(np.array(points, dtype=np.float32), leafsize=32)
    hits = GRID_TREE.query_ball_tree(t, r=COVER_LAT)
    m = np.fromiter((bool(h) for h in hits), bool, len(GRID))
    return m

# ───────────────────────── Simulation API ───────────────────
@app.route('/api/runGreedyStream')
def run_greedy_stream():
    P         = int(request.args.get('n',1000))
    R_block_m = float(request.args.get('r',50))
    MAX       = int(request.args.get('max',1))
    R_eff_m   = 100.0

    CHUNK_G=max(1,P//30); MAX_SWAP1,STOP_FAIL1=1500,150
    TIME_LIMIT,PAIR_SAMP=60,200

    Rblock_lat, Reff_lat = r_lat(R_block_m), r_lat(R_eff_m)

    def neighbours(i,Rm,Rlat):
        lat0,lng0 = LAT[i],LNG[i]
        approx = PT_TREE.query_ball_point([lat0,lng0], r=Rlat)
        if len(approx)==1: return approx
        band=r_lng(Rm,lat0)
        return [j for j in approx if abs(LNG[j]-lng0)<=band and haversine(lat0,lng0,LAT[j],LNG[j])<=Rm]

    def violates(idx,sset):
        return any(haversine(LAT[idx],LNG[idx],LAT[s],LNG[s])<=R_block_m and idx!=s for s in sset)

    assigned=np.zeros(N_PT,np.int32); bins=np.zeros(N_PT,np.int16)
    blocked=np.zeros(N_PT,bool); sel=[]; sel_set=set()

    def add(i):
        if bins[i]==0: sel.append(i); sel_set.add(i)
        bins[i]+=1; assigned[i]+=1
        for k in neighbours(i,R_eff_m,Reff_lat):
            if k!=i: assigned[k]+=1
        for k in neighbours(i,R_block_m,Rblock_lat):
            if k!=i: blocked[k]=True

    def rem(i):
        bins[i]-=1; assigned[i]-=1
        for k in neighbours(i,R_eff_m,Reff_lat):
            if k!=i: assigned[k]-=1
        if bins[i]==0:
            sel.remove(i); sel_set.discard(i)
            for k in neighbours(i,R_block_m,Rblock_lat):
                if k!=i:
                    blocked[k]=any(haversine(LAT[k],LNG[k],LAT[s],LNG[s])<=R_block_m and k!=s for s in sel_set)

    def greedy():
        for i in range(P):
            valid=(~blocked)&(bins<MAX)&cand_mask
            if not valid.any(): break
            gains=np.where(valid, W/(assigned+1), -1)
            j=int(gains.argmax());
            if gains[j]<=0: break
            add(j)
            if (i+1)%CHUNK_G==0 or i==P-1:
                yield f"event: progress\ndata: {int((i+1)/P*70)}\n\n"

    def g(idx):
        nb=neighbours(idx,R_eff_m,Reff_lat)
        return float(np.sum(W[nb]/(assigned[nb]+1)))

    def swap1():
        fail=0
        for s in range(MAX_SWAP1):
            o=random.choice(sel); n=random.choice(cand_idx)
            if n in sel_set or violates(n, sel_set-{o}): continue
            if g(n)-g(o)<=0:
                fail+=1
                if fail>=STOP_FAIL1: break
                continue
            rem(o); add(n); fail=0
            if s%(MAX_SWAP1//25 or 1)==0:
                yield f"event: progress\ndata: {70+int(s/MAX_SWAP1*20)}\n\n"

    def swap2():
        best=float(np.sum(W*assigned/(assigned+1.0)))
        t0=time.time(); improved=True
        while improved and time.time()-t0<TIME_LIMIT:
            improved=False
            sel_pairs  = random.sample(list(itertools.combinations(list(sel_set),2)), k=min(PAIR_SAMP,len(sel_set)*(len(sel_set)-1)//2))
            cand_pairs = random.sample(list(itertools.combinations(cand_idx,2)), PAIR_SAMP)
            for (a,b) in sel_pairs:
                for (c,d) in cand_pairs:
                    if c in sel_set or d in sel_set: continue
                    newset=sel_set-{a,b}|{c,d}
                    if violates(c,newset) or violates(d,newset): continue
                    if g(c)+g(d)-g(a)-g(b)<=0: continue
                    rem(a); rem(b); add(c); add(d)
                    new=float(np.sum(W*assigned/(assigned+1.0)))
                    if new>best: best=new; improved=True; break
                    rem(c); rem(d); add(a); add(b)
                if improved: break
        yield 'event: progress\ndata: 99\n\n'

    def event_stream():
        yield ': connected\n\n'
        yield from greedy(); yield from swap1(); yield from swap2()

        total_red=int(np.sum(W*assigned/(assigned+1)))
        total_bins=int(bins[list(sel_set)].sum())
        result={'selected':[int(i) for i in sel_set], 'bins':bins.tolist(), 'totalReduction':total_red}
        yield f"event: result\ndata: {json.dumps(result)}\n\n"

        # ─ coverage diff 생성 & 방송 ─
                # ─ coverage diff 생성 & 방송 ─
        real_pts = [(float(LAT[i]), float(LNG[i])) for i in np.where(bins > 0)[0]]
        sim_pts  = [(float(LAT[i]), float(LNG[i])) for i in sel_set]

        cov_real = coverage_mask(real_pts)        # bool[|GRID|]
        cov_sim  = coverage_mask(sim_pts)
        diff_arr = cov_sim.astype(np.int8) - cov_real.astype(np.int8)   # -1/0/+1

        # JSON 저장
        GRID_out = GRID.assign(diff=diff_arr)
        diff_path = os.path.join(MAPDIR, 'coverage_diff.json')
        GRID_out[['lat', 'lng', 'diff']].to_json(diff_path, orient='records')

        # overlay + 플래그
        overlay = {
            'points': [
                {'lat': float(LAT[i]), 'lng': float(LNG[i]), 'cnt': int(bins[i])}
                for i in sel_set
            ],
            'installedBins': total_bins,
            'reductionRate': round(total_red / TOTAL_WASTE * 100, 1),
            'coverageReady': True                        #  ← 새 플래그
        }
        publish_overlay(overlay)

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
