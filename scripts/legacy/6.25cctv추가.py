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
GRID  = ALL[['lat', 'lng']].copy()
GRID_LATLNG = GRID.values.astype(np.float32)
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
    R=6_371_000.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ, dλ = radians(lat2-lat1), radians(lng2-lng1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def r_lat(Rm):  return Rm/111_320.0

def r_lng(Rm, φ): return Rm/(111_320.0*cos(radians(φ)))

# ───────────────────────── Flask ────────────────────────────
app = Flask(__name__, static_folder=STATIC, static_url_path='')

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

# 정적 라우트
@app.route('/')
def index(): return send_from_directory(STATIC, '6.15클러스터링개조버전.html')

@app.route('/map/<path:p>')
def maps(p): return send_from_directory(MAPDIR, p)

@app.route('/gangnam')
def gangnam(): return send_from_directory(STATIC, '6.25gangnamcctv추가버전.html')

@app.route('/api/markers')
def api_markers():
    with open(os.path.join(MAPDIR,'gangnam_markers.geojson'),encoding='utf-8') as f:
        return jsonify(json.load(f))

# CCTV 위치 JSON 서빙
@app.route('/api/cctvs')
def api_cctvs():
    with open(os.path.join(MAPDIR,'cctvs.json'), encoding='utf-8') as f:
        return jsonify(json.load(f))

# ───────────────────────── Coverage helper ──────────────────
COVER_R   = 100.0
COVER_LAT = r_lat(COVER_R)

def coverage_mask(points):
    if not points:
        return np.zeros(len(GRID), bool)
    t = cKDTree(np.array(points, dtype=np.float32), leafsize=32)
    hits = GRID_TREE.query_ball_tree(t, r=COVER_LAT)
    return np.fromiter((bool(h) for h in hits), bool, len(GRID))

# ───────────────────────── Simulation API ───────────────────
@app.route('/api/runGreedyStream')
def run_greedy_stream():
    P         = int(request.args.get('n',1000))
    R_block_m = float(request.args.get('r',50))
    MAX       = int(request.args.get('max',1))
    R_eff_m   = 100.0

    CHUNK_G,max1,stop1 = max(1,P//30),1500,150
    TL,PS = 60,200
    Rblock_lat,Reff_lat = r_lat(R_block_m), r_lat(R_eff_m)

    def neighbours(i,Rm,Rlat):
        lat0,lng0 = LAT[i],LNG[i]
        approx = PT_TREE.query_ball_point([lat0,lng0], r=Rlat)
        if len(approx)==1: return approx
        band=r_lng(Rm,lat0)
        return [j for j in approx if abs(LNG[j]-lng0)<=band and haversine(lat0,lng0,LAT[j],LNG[j])<=Rm]
    def violates(i,s): return any(haversine(LAT[i],LNG[i],LAT[s0],LNG[s0])<=R_block_m and i!=s0 for s0 in s)

    assigned = np.zeros(N_PT,np.int32); bins = np.zeros(N_PT,np.int16)
    blocked  = np.zeros(N_PT,bool); sel=[]; sel_set=set()
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
                    blocked[k]=any(haversine(LAT[k],LNG[k],LAT[s0],LNG[s0])<=R_block_m and k!=s0 for s0 in sel_set)

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

    def gain(i): return float(np.sum(W[neighbours(i,R_eff_m,Reff_lat)]/(assigned[neighbours(i,R_eff_m,Reff_lat)]+1)))
    def swap1():
        f=0
        for s0 in range(max1):
            o=random.choice(sel); n=random.choice(cand_idx)
            if n in sel_set or violates(n, sel_set-{o}): continue
            if gain(n)-gain(o)<=0:
                f+=1
                if f>=stop1: break
                continue
            rem(o); add(n); f=0
            if s0%(max1//25 or 1)==0:
                yield f"event: progress\ndata: {70+int(s0/max1*20)}\n\n"

    def swap2():
        best=float(np.sum(W*assigned/(assigned+1.0)))
        t0=time.time(); imp=True
        while imp and time.time()-t0<TL:
            imp=False
            sp=random.sample(list(itertools.combinations(sel_set,2)), k=min(PS,len(sel_set)*(len(sel_set)-1)//2))
            cp=random.sample(list(itertools.combinations(cand_idx,2)), PS)
            for a,b in sp:
                for c,d in cp:
                    if c in sel_set or d in sel_set: continue
                    ns=sel_set-{a,b}|{c,d}
                    if violates(c,ns) or violates(d,ns): continue
                    if gain(c)+gain(d)-gain(a)-gain(b)<=0: continue
                    rem(a); rem(b); add(c); add(d)
                    new=float(np.sum(W*assigned/(assigned+1.0)))
                    if new>best: best=new; imp=True; break
                    rem(c); rem(d); add(a); add(b)
                if imp: break
        yield 'event: progress\ndata: 99\n\n'

    def event_stream():
        yield ': connected\n\n'
        yield from greedy(); yield from swap1(); yield from swap2()

        total_red=int(np.sum(W*assigned/(assigned+1)))
        total_bins=int(bins[list(sel_set)].sum())
        yield f"event: result\ndata: {json.dumps({'selected':[int(i) for i in sel_set],'bins':bins.tolist(),'totalReduction':total_red})}\n\n"

        # coverage diff
        real_pts=[(float(LAT[i]),float(LNG[i])) for i in np.where(bins>0)[0]]
        sim_pts=[(float(LAT[i]),float(LNG[i])) for i in sel_set]
        cov_real=coverage_mask(real_pts)
        cov_sim =coverage_mask(sim_pts)
        diff_arr=cov_sim.astype(int)-cov_real.astype(int)
        GRID_out=GRID.assign(diff=diff_arr)
        GRID_out[['lat','lng','diff']].to_json(os.path.join(MAPDIR,'coverage_diff.json'), orient='records')

        overlay={'points':[{'lat':float(LAT[i]),'lng':float(LNG[i]),'cnt':int(bins[i])} for i in sel_set],
                 'installedBins': total_bins,
                 'reductionRate': round(total_red/TOTAL_WASTE*100,1),
                 'coverageReady': True}
        publish_overlay(overlay)
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
