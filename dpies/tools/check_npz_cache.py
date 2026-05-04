#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ACTION_NAMES = {
    0: 'keep', 1: 'stop', 2: 'proceed', 3: 'lane_change_left',
    4: 'lane_change_right', 5: 'merge', 6: 'nudge_left', 7: 'nudge_right', 8: 'creep'
}
EVIDENCE_NAMES = {0:'dynamic_agent', 1:'conflict_point', 2:'gap', 3:'map_rule', 4:'low_ttc_risk', 5:'padding'}
RULE_NAMES = {1:'stop_line',2:'crosswalk',3:'lane_boundary',4:'traffic_light_red',5:'drivable_area',6:'speed_limit',7:'route_deviation',8:'intersection',9:'lane_connector'}

def _json(v):
    if isinstance(v, np.ndarray):
        v = v.item() if v.shape == () else v.tolist()
    if isinstance(v, bytes):
        v = v.decode('utf-8')
    if not isinstance(v, str):
        v = str(v)
    try:
        return json.loads(v)
    except Exception:
        return {}

def _mask(z, *names, n=0):
    for name in names:
        if name in z.files:
            return np.asarray(z[name]).astype(bool)
    return np.ones((n,), dtype=bool)

def _min_ade_fde(actions, action_mask, logged):
    if actions.size == 0 or logged.size == 0 or not action_mask.any():
        return math.inf, math.inf
    tgt = logged[: actions.shape[1], :2]
    if len(tgt) == 0:
        return math.inf, math.inf
    dist = np.linalg.norm(actions[action_mask, :len(tgt), :2] - tgt[None, :, :], axis=-1)
    return float(dist.mean(axis=1).min()), float(dist[:, -1].min())

def check_file(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = _json(z['metadata_json']) if 'metadata_json' in z.files else {}
    ev_meta = _json(z['evidence_metadata_json']) if 'evidence_metadata_json' in z.files else []
    route = _json(z['route_info_json']) if 'route_info_json' in z.files else {}

    actions = np.asarray(z['actions']) if 'actions' in z.files else np.zeros((0,0,0), dtype=np.float32)
    action_mask = _mask(z, 'action_mask', 'action_valid_mask', n=actions.shape[0])
    evidence_mask = _mask(z, 'evidence_mask', n=np.asarray(z['evidence_features']).shape[0] if 'evidence_features' in z.files else 0)
    e_type = np.asarray(z['evidence_type']).astype(int) if 'evidence_type' in z.files else np.zeros((len(evidence_mask),), dtype=int)
    action_meta = np.asarray(z['action_meta']) if 'action_meta' in z.files else np.zeros((actions.shape[0], 8), dtype=np.float32)
    teacher = np.asarray(z['teacher_cost']) if 'teacher_cost' in z.files else np.zeros((actions.shape[0],), dtype=np.float32)
    signed = np.asarray(z['signed_evidence_label']) if 'signed_evidence_label' in z.files else np.zeros((len(evidence_mask), actions.shape[0], actions.shape[0]), dtype=np.float32)
    smask = np.asarray(z['signed_evidence_mask']).astype(bool) if 'signed_evidence_mask' in z.files else np.zeros_like(signed, dtype=bool)
    q = np.asarray(z['geometry_query']) if 'geometry_query' in z.files else np.zeros((len(evidence_mask), actions.shape[0], 24), dtype=np.float32)
    logged = np.asarray(z['logged_ego_future']) if 'logged_ego_future' in z.files else np.zeros((0, 9), dtype=np.float32)

    layers = Counter(); rules = Counter(); meta_type = Counter()
    has = Counter()
    for m in ev_meta if isinstance(ev_meta, list) else []:
        if not isinstance(m, dict):
            continue
        layer = str(m.get('layer',''))
        layers[layer] += 1
        meta_type[str(m.get('type',''))] += 1
        code = m.get('rule_code', None)
        if code is not None:
            rules[RULE_NAMES.get(int(code), str(code))] += 1
        has['drivable'] += int(layer == 'DRIVABLE_AREA_UNION')
        has['route'] += int(layer == 'ROUTE_CORRIDOR')
        has['crosswalk'] += int('CROSSWALK' in layer)
        has['stop'] += int('STOP' in layer)
        has['traffic_light'] += int('TRAFFIC_LIGHT' in layer)
        has['speed_limit'] += int(layer == 'SPEED_LIMIT')

    valid_e = evidence_mask & (e_type != 5)
    et_counts = Counter(EVIDENCE_NAMES.get(int(t), str(int(t))) for t in e_type[valid_e])
    mode_counts = Counter(ACTION_NAMES.get(int(m), str(int(m))) for m in action_meta[action_mask, 0].astype(int))

    valid_pair = action_mask[:, None] & action_mask[None, :] & (~np.eye(len(action_mask), dtype=bool))
    signed_valid = valid_e[:, None, None] & valid_pair[None, :, :]
    signed_nonzero = float((np.abs(signed[signed_valid]) > 1e-6).mean()) if signed_valid.any() else 0.0
    signed_active = float(smask[signed_valid].mean()) if signed_valid.any() else 0.0
    antisym = None
    if signed_valid.any():
        ssum = signed + np.swapaxes(signed, -1, -2)
        antisym = float(np.max(np.abs(ssum[signed_valid])))
    q_valid = q[..., 23] if q.ndim == 3 and q.shape[-1] >= 24 else np.zeros(q.shape[:2], dtype=np.float32)
    q_valid_ratio = float(q_valid[evidence_mask[:, None] & action_mask[None, :]].mean()) if evidence_mask.any() and action_mask.any() else 0.0
    q_finite = bool(np.isfinite(q).all())
    teacher_valid = teacher[action_mask]
    tc_std = float(np.std(teacher_valid)) if teacher_valid.size else math.nan
    tc_span = float(np.max(teacher_valid) - np.min(teacher_valid)) if teacher_valid.size else math.nan
    ade, fde = _min_ade_fde(actions, action_mask, logged)
    lf_disp = float(np.linalg.norm(logged[-1, :2])) if logged.size and len(logged) else math.nan
    action_max_x = float(np.max(actions[action_mask, :, 0])) if action_mask.any() else math.nan
    action_max_abs_y = float(np.max(np.abs(actions[action_mask, :, 1]))) if action_mask.any() else math.nan

    return {
        'file': path.name,
        'scenario_id': meta.get('scenario_id',''),
        'map_success': bool(meta.get('map_success', False)),
        'map_error': str(meta.get('map_error','')),
        'action_valid_count': int(action_mask.sum()),
        'evidence_valid_count': int(valid_e.sum()),
        'evidence_type_counts': dict(et_counts),
        'action_mode_counts': dict(mode_counts),
        'map_layer_counts': dict(layers),
        'rule_code_counts': dict(rules),
        'has_drivable_area_union': has['drivable'] > 0,
        'has_route_corridor': has['route'] > 0,
        'has_crosswalk': has['crosswalk'] > 0,
        'has_stop_line': has['stop'] > 0,
        'has_traffic_light': has['traffic_light'] > 0,
        'has_speed_limit': has['speed_limit'] > 0,
        'route_polygons': len(route.get('route_polygons', [])) if isinstance(route, dict) else 0,
        'route_polylines': len(route.get('route_polylines', [])) if isinstance(route, dict) else 0,
        'teacher_cost_finite': bool(np.isfinite(teacher).all()),
        'teacher_cost_std': tc_std,
        'teacher_cost_span': tc_span,
        'geometry_query_finite': q_finite,
        'geometry_query_valid_ratio': q_valid_ratio,
        'signed_evidence_finite': bool(np.isfinite(signed).all()),
        'signed_nonzero_ratio': signed_nonzero,
        'signed_active_ratio': signed_active,
        'signed_antisym_max_abs_error': antisym,
        'rival_positive_ratio': float(np.asarray(z['rival_label']).astype(bool)[valid_pair].mean()) if 'rival_label' in z.files and valid_pair.any() else 0.0,
        'oracle_action_index': int(np.asarray(z['oracle_action_index']).item()) if 'oracle_action_index' in z.files else -1,
        'min_ade': ade,
        'min_fde': fde,
        'logged_future_final_distance': lf_disp,
        'action_max_x': action_max_x,
        'action_max_abs_y': action_max_abs_y,
        'critical_bad': bool((not bool(meta.get('map_success', False))) or action_mask.sum() < 8 or not np.isfinite(teacher).all() or not q_finite or not np.isfinite(signed).all() or ade > 25.0),
        'warnings': []
    }

def pct(vals, p):
    if not vals: return math.nan
    return float(np.percentile(np.asarray(vals, dtype=float), p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--show-files', type=int, default=5)
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()
    files = sorted(Path(args.cache_dir).rglob("*.npz"))
    if args.limit > 0: files = files[:args.limit]
    rows = [check_file(f) for f in files]
    print('cache_dir:', args.cache_dir)
    print('checked_files:', len(rows))
    keys_bool = ['map_success','has_drivable_area_union','has_route_corridor','has_crosswalk','has_stop_line','has_traffic_light','has_speed_limit','teacher_cost_finite','geometry_query_finite','signed_evidence_finite','critical_bad']
    for k in keys_bool:
        c = sum(int(r[k]) for r in rows)
        print(f'{k}: {c}/{len(rows)} = {c/max(len(rows),1):.3f}')
    for k in ['action_valid_count','evidence_valid_count','min_ade','min_fde','logged_future_final_distance','teacher_cost_std','teacher_cost_span','geometry_query_valid_ratio','signed_nonzero_ratio','signed_active_ratio','rival_positive_ratio','signed_antisym_max_abs_error']:
        vals = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        print(f'{k}: mean={np.mean(vals) if vals else math.nan:.4g} p50={pct(vals,50):.4g} p90={pct(vals,90):.4g} max={max(vals) if vals else math.nan:.4g}')
    et = Counter(); modes = Counter(); layers = Counter(); rules = Counter()
    for r in rows:
        et.update(r['evidence_type_counts']); modes.update(r['action_mode_counts']); layers.update(r['map_layer_counts']); rules.update(r['rule_code_counts'])
    print('\n=== evidence types ===')
    for k,v in et.most_common(): print(f'{v:8d} {k}')
    print('\n=== action modes ===')
    for k,v in modes.most_common(): print(f'{v:8d} {k}')
    print('\n=== map layers ===')
    for k,v in layers.most_common(30): print(f'{v:8d} {repr(k)}')
    print('\n=== rule codes ===')
    for k,v in rules.most_common(30): print(f'{v:8d} {k}')
    print('\n=== examples ===')
    for r in rows[:args.show_files]:
        print(json.dumps(r, ensure_ascii=False, indent=2)[:4000])
    bad = [r for r in rows if r['critical_bad']]
    print('\n=== critical_bad examples ===')
    for r in bad[:20]:
        print(r['file'], 'ade=', r['min_ade'], 'fde=', r['min_fde'], 'actions=', r['action_valid_count'], 'map=', r['map_success'])
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()