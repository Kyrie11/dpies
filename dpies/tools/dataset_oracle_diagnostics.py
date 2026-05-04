import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='Dataset oracle diagnostics')
parser.add_argument('--CACHE_DIR', type=str, required=True,
                   help='Path to the cache directory containing npz files')
args = parser.parse_args()

# 使用命令行参数中的 CACHE_DIR
cache = Path(args.CACHE_DIR)
if not cache.exists():
    raise FileNotFoundError(f"CACHE_DIR not found: {cache}")

files = sorted(cache.rglob("*.npz"))
print("CACHE_DIR:", cache)
print("num npz:", len(files))

# Inspect first few files
print("\nFIRST FILE KEYS")
for p in files[:3]:
    try:
        d = np.load(p, allow_pickle=True)
        print("file:", p.name)
        print("keys:", sorted(list(d.keys())))
    except Exception as e:
        print("bad file:", p, e)

rows = []
bad = []

component_names_12 = [
    "ade", "fde", "progress_norm", "speed_over",
    "accel_cost", "jerk_cost", "curv_cost",
    "future_collision", "future_proximity",
    "local_sum", "global_without_local", "total",
]

for idx, p in enumerate(files):
    try:
        d = np.load(p, allow_pickle=True)

        required = ["actions", "action_mask", "teacher_cost"]
        missing = [k for k in required if k not in d]
        if missing:
            bad.append((p.name, f"missing {missing}"))
            continue

        actions = d["actions"]
        action_mask = d["action_mask"].astype(bool)
        teacher_cost = d["teacher_cost"].astype(np.float32)

        if actions.ndim != 3 or actions.shape[-1] < 4:
            bad.append((p.name, f"bad actions shape {actions.shape}"))
            continue

        valid = np.where(action_mask)[0]
        if len(valid) == 0:
            bad.append((p.name, "no valid action"))
            continue

        if "oracle_action_index" in d:
            oracle = int(d["oracle_action_index"])
        else:
            tmp = teacher_cost.copy()
            tmp[~action_mask] = np.inf
            oracle = int(np.argmin(tmp))

        if oracle < 0 or oracle >= len(action_mask) or not action_mask[oracle]:
            tmp = teacher_cost.copy()
            tmp[~action_mask] = np.inf
            oracle = int(np.argmin(tmp))

        progress = actions[:, -1, 0].astype(np.float32)
        final_speed = actions[:, -1, 3].astype(np.float32)

        max_progress_idx = int(valid[np.argmax(progress[valid])])
        min_cost_idx = int(valid[np.argmin(teacher_cost[valid])])

        row = {
            "file": p.name,
            "valid_action_count": int(action_mask.sum()),
            "oracle": oracle,
            "min_cost_idx": min_cost_idx,
            "oracle_progress": float(progress[oracle]),
            "oracle_final_speed": float(final_speed[oracle]),
            "oracle_teacher_cost": float(teacher_cost[oracle]),
            "max_progress": float(progress[max_progress_idx]),
            "max_progress_final_speed": float(final_speed[max_progress_idx]),
            "max_progress_teacher_cost": float(teacher_cost[max_progress_idx]),
            "teacher_cost_min": float(np.min(teacher_cost[valid])),
            "teacher_cost_p50": float(np.percentile(teacher_cost[valid], 50)),
            "teacher_cost_max": float(np.max(teacher_cost[valid])),
            "oracle_is_near_stop": float(final_speed[oracle] < 0.5),
            "oracle_low_progress_8m": float(progress[oracle] < 8.0),
            "oracle_low_progress_20m": float(progress[oracle] < 20.0),
            "max_progress_minus_oracle": float(progress[max_progress_idx] - progress[oracle]),
            "cost_max_progress_minus_oracle": float(teacher_cost[max_progress_idx] - teacher_cost[oracle]),
        }

        if "action_meta" in d:
            meta = d["action_meta"]
            if meta.ndim == 2 and meta.shape[0] > oracle:
                row["oracle_mode"] = int(meta[oracle, 0])
                row["max_progress_mode"] = int(meta[max_progress_idx, 0])

        if "rival_label" in d:
            rival = d["rival_label"].astype(bool)
            row["rival_label_count"] = int(rival.sum())
            row["rival_label_density"] = float(rival.sum() / max(1, action_mask.sum() * (action_mask.sum() - 1)))

            # oracle-vs-any rival coverage
            row["oracle_rival_out"] = int(rival[oracle].sum())
            row["oracle_rival_in"] = int(rival[:, oracle].sum())

        if "signed_evidence_mask" in d:
            sem = d["signed_evidence_mask"].astype(bool)
            row["signed_evidence_active_count"] = int(sem.sum())
            row["signed_evidence_active_density"] = float(sem.mean())

        if "evidence_mask" in d:
            em = d["evidence_mask"].astype(bool)
            row["evidence_count"] = int(em.sum())

        if "teacher_components" in d:
            comp = d["teacher_components"]
            if comp.ndim == 2 and comp.shape[0] > oracle:
                for j in range(min(comp.shape[1], len(component_names_12))):
                    row[f"oracle_comp_{component_names_12[j]}"] = float(comp[oracle, j])
                    row[f"maxprog_comp_{component_names_12[j]}"] = float(comp[max_progress_idx, j])

        rows.append(row)

    except Exception as e:
        bad.append((p.name, repr(e)))

df = pd.DataFrame(rows)
print("\nGOOD FILES:", len(df))
print("BAD FILES:", len(bad))
if bad[:10]:
    print("BAD EXAMPLES:")
    for x in bad[:10]:
        print(x)

if len(df) == 0:
    print("No valid rows. Likely wrong CACHE_DIR or incompatible npz schema.")
    raise SystemExit(0)

def describe(cols):
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(df[cols].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).T.to_string())

print("\nORACLE / ACTION COVERAGE")
describe([
    "valid_action_count",
    "oracle_progress",
    "oracle_final_speed",
    "max_progress",
    "max_progress_final_speed",
    "max_progress_minus_oracle",
    "cost_max_progress_minus_oracle",
    "oracle_teacher_cost",
    "max_progress_teacher_cost",
    "teacher_cost_min",
    "teacher_cost_p50",
    "teacher_cost_max",
])

print("\nORACLE STOP RATIOS")
for c in ["oracle_is_near_stop", "oracle_low_progress_8m", "oracle_low_progress_20m"]:
    if c in df.columns:
        print(c, "mean=", df[c].mean(), "sum=", int(df[c].sum()), "/", len(df))

if "oracle_mode" in df.columns:
    print("\nORACLE MODE COUNTS")
    print(df["oracle_mode"].value_counts(dropna=False).sort_index())

if "max_progress_mode" in df.columns:
    print("\nMAX PROGRESS MODE COUNTS")
    print(df["max_progress_mode"].value_counts(dropna=False).sort_index())

print("\nRIVAL / EVIDENCE")
describe([
    "rival_label_count",
    "rival_label_density",
    "oracle_rival_out",
    "oracle_rival_in",
    "signed_evidence_active_count",
    "signed_evidence_active_density",
    "evidence_count",
])

comp_cols = [c for c in df.columns if c.startswith("oracle_comp_")]
if comp_cols:
    print("\nORACLE TEACHER COMPONENTS")
    describe(comp_cols)

# Cases where teacher prefers stop/low-progress despite high-progress available
sus = df[
    (df["oracle_final_speed"] < 0.5)
    & (df["max_progress"] >= 20.0)
].copy()
print("\nSUSPECT: oracle near stop while max_progress>=20:", len(sus), "/", len(df))
if len(sus):
    show_cols = [
        "file", "oracle", "oracle_mode", "oracle_progress", "oracle_final_speed",
        "max_progress", "max_progress_final_speed",
        "max_progress_minus_oracle",
        "cost_max_progress_minus_oracle",
        "oracle_teacher_cost", "max_progress_teacher_cost",
    ]
    show_cols = [c for c in show_cols if c in sus.columns]
    print(sus.sort_values("max_progress_minus_oracle", ascending=False)[show_cols].head(30).to_string(index=False))

out = Path("runs/dataset_oracle_diagnostics.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("\nsaved:", out.resolve())
