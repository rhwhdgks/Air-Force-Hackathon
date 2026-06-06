#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, random, time, itertools, numpy as np, pandas as pd
from flask import Flask, Response, request, send_from_directory, stream_with_context
from scipy.spatial import cKDTree
from math import radians, sin, cos, atan2, sqrt

# ───────────────────────── 경로 · 로드 ─────────────────────────
ROOT   = os.path.dirname(__file__)
MAPDIR = os.path.join(ROOT, 'map')
STATIC = os.path.join(ROOT, 'static')

ALL  = pd.read_json(os.path.join(MAPDIR, 'GangNam_garbage_pred_10m.json'))
CAND = pd.read_json(os.path.join(MAPDIR, 'candidates_kmeans_10000_1.json'))

LAT = ALL.lat.values.astype(np.float32)
LNG = ALL.lng.values.astype(np.float32)
W   = ALL.waste_estimate.values.astype(np.float32)
N_PT = len(ALL)

cand_idx  = CAND.origin_idx.values.astype(np.int32)
cand_mask = np.zeros(N_PT, bool); cand_mask[cand_idx] = True

tree = cKDTree(np.c_[LAT, LNG], leafsize=64)

# ───────────────────────── 지오 헬퍼 ──────────────────────────
def haversine(lat1,lng1,lat2,lng2):
    R=6_371_000.0
    φ1,φ2 = radians(lat1),radians(lat2)
    dφ,dλ = radians(lat2-lat1),radians(lng2-lng1)
    a=sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2*R*atan2(sqrt(a),sqrt(1-a))

def r_lat(Rm):      return Rm/111_320.0
def r_lng(Rm,φ):    return Rm/(111_320.0*cos(radians(φ)))

# ───────────────────────── Flask ────────────────────────────
app = Flask(__name__, static_folder=STATIC, static_url_path='')

@app.route('/')
def index(): return send_from_directory(STATIC,'6.15클러스터링개조버전.html')
@app.route('/map/<path:p>')          # 정적 지도 파일
def maps(p): return send_from_directory(MAPDIR,p)

# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
@app.route('/api/runGreedyStream')
def run_greedy_stream():
    P         = int(request.args.get('n',   1000))
    R_block_m = float(request.args.get('r', 50.0))
    MAX       = int(request.args.get('max', 1))
    R_eff_m   = 100.0

    CHUNK_G    = max(1, P // 30)
    MAX_SWAP1  = 1500
    STOP_FAIL1 = 150
    TIME_LIMIT = 60
    PAIR_SAMP  = 200

    Rblock_lat = r_lat(R_block_m)
    Reff_lat   = r_lat(R_eff_m)

    # ── 인접 검색 ────────────────────────────────────────────────
    def neighbours(idx, Rm, Rlat):
        lat0, lng0 = LAT[idx], LNG[idx]
        approx = tree.query_ball_point([lat0, lng0], r=Rlat)
        if len(approx) == 1:
            return approx
        band = r_lng(Rm, lat0)
        return [
            j for j in approx
            if abs(LNG[j]-lng0) <= band and
               haversine(lat0, lng0, LAT[j], LNG[j]) <= Rm
        ]

    def violates_block(idx, sel_set_local):
        for s in sel_set_local:
            if haversine(LAT[idx], LNG[idx], LAT[s], LNG[s]) <= R_block_m and idx != s:
                return True
        return False

    # ── 상태 배열 ────────────────────────────────────────────────
    assigned = np.zeros(N_PT, np.int32)
    bins     = np.zeros(N_PT, np.int16)
    blocked  = np.zeros(N_PT, bool)
    sel      = []
    sel_set  = set()

    # ── 헬퍼 ────────────────────────────────────────────────────
    def add_bin(idx):
        if bins[idx] == 0:
            sel.append(idx)
            sel_set.add(idx)
        bins[idx]     += 1
        assigned[idx] += 1
        for k in neighbours(idx, R_eff_m, Reff_lat):
            if k != idx:
                assigned[k] += 1
        for k in neighbours(idx, R_block_m, Rblock_lat):
            if k != idx:
                blocked[k] = True

    def remove_bin(idx):
        bins[idx]     -= 1
        assigned[idx] -= 1
        for k in neighbours(idx, R_eff_m, Reff_lat):
            if k != idx:
                assigned[k] -= 1
        if bins[idx] == 0:
            sel.remove(idx)
            sel_set.discard(idx)
            for k in neighbours(idx, R_block_m, Rblock_lat):
                if k != idx:
                    blocked[k] = any(
                        haversine(LAT[k], LNG[k], LAT[s], LNG[s]) <= R_block_m and k != s
                        for s in sel_set
                    )

    # ── 1) Greedy ───────────────────────────────────────────────
    def greedy_phase():
        for i in range(P):
            valid = (~blocked) & (bins < MAX) & cand_mask
            if not valid.any():
                break
            gains = np.where(valid, W / (assigned + 1), -1)
            j = int(gains.argmax())
            if gains[j] <= 0:
                break
            add_bin(j)

            if (i + 1) % CHUNK_G == 0 or i == P - 1:
                pct = int((i + 1) / P * 70)
                yield f'event: progress\ndata: {pct}\n\n'

    # ── 2) 1-swap ───────────────────────────────────────────────
    def gain(idx):
        nb = neighbours(idx, R_eff_m, Reff_lat)
        return np.sum(W[nb] / (assigned[nb] + 1))

    def one_swap_phase():
        fail = 0
        for s in range(MAX_SWAP1):
            out = random.choice(sel)
            inn = random.choice(cand_idx)
            if inn in sel_set:
                continue
            if violates_block(inn, sel_set - {out}):
                continue
            if gain(inn) - gain(out) <= 0:
                fail += 1
                if fail >= STOP_FAIL1:
                    break
                continue
            remove_bin(out)
            add_bin(inn)
            fail = 0
            if s % (MAX_SWAP1 // 25 or 1) == 0:
                pct = 70 + int(s / MAX_SWAP1 * 20)
                yield f'event: progress\ndata: {pct}\n\n'

    # ── 3) 2-swap ───────────────────────────────────────────────
    def apply_swap(out_pair, in_pair, sign):
        if sign == +1:
            for o in out_pair:
                remove_bin(o)
            for n in in_pair:
                add_bin(n)
        else:
            for o in out_pair:
                add_bin(o)
            for n in in_pair:
                remove_bin(n)

    def two_swap_phase():
        best_red = np.sum(W * assigned / (assigned + 1.0))
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < TIME_LIMIT:
            improved = False
            sel_pairs = random.sample(
                list(itertools.combinations(list(sel_set), 2)),
                k=min(PAIR_SAMP, len(sel_set)*(len(sel_set)-1)//2)
            )
            cand_pairs = random.sample(
                list(itertools.combinations(cand_idx, 2)), PAIR_SAMP
            )

            for (a, b) in sel_pairs:
                for (c, d) in cand_pairs:
                    if c in sel_set or d in sel_set:
                        continue
                    new_sel = sel_set - {a, b} | {c, d}
                    if violates_block(c, new_sel) or violates_block(d, new_sel):
                        continue
                    if gain(c) + gain(d) - gain(a) - gain(b) <= 0:
                        continue
                    apply_swap((a, b), (c, d), +1)
                    new_red = np.sum(W * assigned / (assigned + 1.0))
                    if new_red > best_red:
                        best_red = new_red
                        improved = True
                        break
                    else:
                        apply_swap((a, b), (c, d), -1)
                if improved:
                    break
        yield 'event: progress\ndata: 99\n\n'

    # ── SSE 스트림 ──────────────────────────────────────────────
    def event_stream():
        yield ': connected\n\n'
        yield from greedy_phase()
        yield from one_swap_phase()
        yield from two_swap_phase()

        total_reduction = int(np.sum(W * assigned / (assigned + 1)))
        payload = {
            'selected': [int(i) for i in sel_set],
            'bins':     bins.tolist(),
            'totalReduction': total_reduction
        }
        yield f'event: result\ndata: {json.dumps(payload)}\n\n'

    return Response(event_stream(), mimetype='text/event-stream')
# ────────────────────────────────────────────────────────────────




if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
