import json
import pandas as pd
from pathlib import Path

path = Path("runs/closed_loop_action_debug.jsonl")
if not path.exists():
    raise FileNotFoundError(f"not found: {path.resolve()}")

rows = []
bad_lines = 0

with path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            bad_lines += 1
            print(f"[bad json line {i}] {e}")

df = pd.json_normalize(rows, sep=".")

print("=" * 100)
print("BASIC")
print("=" * 100)
print("file:", path.resolve())
print("steps:", len(df))
print("bad_lines:", bad_lines)
print("columns:")
for c in df.columns:
    print("  -", c)

def show_value_counts(col, top=30):
    print("\n" + "=" * 100)
    print(f"VALUE COUNTS: {col}")
    print("=" * 100)
    if col not in df.columns:
        print(f"[missing column] {col}")
        return
    print(df[col].fillna("<NA>").value_counts(dropna=False).head(top))

def show_numeric(col):
    print("\n" + "=" * 100)
    print(f"NUMERIC SUMMARY: {col}")
    print("=" * 100)
    if col not in df.columns:
        print(f"[missing column] {col}")
        return
    s = pd.to_numeric(df[col], errors="coerce")
    print("count:", int(s.count()), "missing:", int(s.isna().sum()))
    if s.count() == 0:
        return
    print("mean:", s.mean())
    print("std:", s.std())
    print("min:", s.min())
    print("p01:", s.quantile(0.01))
    print("p05:", s.quantile(0.05))
    print("p10:", s.quantile(0.10))
    print("p25:", s.quantile(0.25))
    print("p50:", s.quantile(0.50))
    print("p75:", s.quantile(0.75))
    print("p90:", s.quantile(0.90))
    print("p95:", s.quantile(0.95))
    print("p99:", s.quantile(0.99))
    print("max:", s.max())

# --------------------------------------------------------------------------------------
# Common categorical / status fields
# --------------------------------------------------------------------------------------
for col in [
    "selected_mode",
    "rerank_reason",
    "fallback",
    "fallback_reason",
    "debug.fallback_reason",
    "debug.map_success",
    "debug.map_error",
    "debug.route_info.source",
    "debug.route_info.reason",
]:
    show_value_counts(col)

# --------------------------------------------------------------------------------------
# Common numeric fields from your planner jsonl
# --------------------------------------------------------------------------------------
numeric_cols = [
    "selected_action",
    "raw_model_action",
    "selected_progress",
    "min_progress",
    "selected_final_speed",
    "selected_comfort_violation",
    "q_selected",
    "selected_rerank_score",
    "valid_action_count",
    "evidence_count",
    "debug.valid_action_count",
    "debug.evidence_count",
    "debug.route_roadblocks",
    "debug.traffic_lights",
    "debug.input_s",
    "debug.model_select_s",
]

# Add every timing field automatically, e.g. debug.timings.action_gen_s
numeric_cols += [c for c in df.columns if c.startswith("debug.timings.")]

# Deduplicate while preserving order
seen = set()
numeric_cols = [c for c in numeric_cols if not (c in seen or seen.add(c))]

for col in numeric_cols:
    show_numeric(col)

# --------------------------------------------------------------------------------------
# Derived diagnostics
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("DERIVED DIAGNOSTICS")
print("=" * 100)

if "selected_progress" in df.columns and "min_progress" in df.columns:
    df["progress_gap"] = pd.to_numeric(df["selected_progress"], errors="coerce") - pd.to_numeric(df["min_progress"], errors="coerce")
    show_numeric("progress_gap")

    bad_progress = df[df["progress_gap"] < 0]
    print("\nprogress_gap < 0 count:", len(bad_progress))
    if len(df) > 0:
        print("progress_gap < 0 ratio:", len(bad_progress) / len(df))

if "raw_model_action" in df.columns and "selected_action" in df.columns:
    df["reranked_changed_action"] = (
        pd.to_numeric(df["raw_model_action"], errors="coerce")
        != pd.to_numeric(df["selected_action"], errors="coerce")
    )
    show_value_counts("reranked_changed_action")

if "selected_comfort_violation" in df.columns:
    comfort = pd.to_numeric(df["selected_comfort_violation"], errors="coerce")
    print("\ncomfort violation >= 0.5 count:", int((comfort >= 0.5).sum()))
    if comfort.notna().sum() > 0:
        print("comfort violation >= 0.5 ratio among non-null:", float((comfort >= 0.5).sum() / comfort.notna().sum()))

if "selected_final_speed" in df.columns:
    speed = pd.to_numeric(df["selected_final_speed"], errors="coerce")
    print("\nnear stop final speed < 0.5 count:", int((speed < 0.5).sum()))
    if speed.notna().sum() > 0:
        print("near stop final speed < 0.5 ratio among non-null:", float((speed < 0.5).sum() / speed.notna().sum()))

# --------------------------------------------------------------------------------------
# Grouped summaries
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("GROUPED SUMMARY BY rerank_reason")
print("=" * 100)

if "rerank_reason" in df.columns:
    agg_cols = []
    for c in [
        "selected_progress",
        "min_progress",
        "progress_gap",
        "selected_final_speed",
        "selected_comfort_violation",
        "q_selected",
        "selected_rerank_score",
        "valid_action_count",
        "evidence_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg_cols.append(c)

    if agg_cols:
        print(
            df.groupby("rerank_reason", dropna=False)[agg_cols]
              .agg(["count", "mean", "min", "median", "max"])
              .to_string()
        )

print("\n" + "=" * 100)
print("GROUPED SUMMARY BY selected_mode")
print("=" * 100)

if "selected_mode" in df.columns:
    agg_cols = []
    for c in [
        "selected_progress",
        "min_progress",
        "progress_gap",
        "selected_final_speed",
        "selected_comfort_violation",
        "q_selected",
        "selected_rerank_score",
        "valid_action_count",
        "evidence_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg_cols.append(c)

    if agg_cols:
        print(
            df.groupby("selected_mode", dropna=False)[agg_cols]
              .agg(["count", "mean", "min", "median", "max"])
              .to_string()
        )

# --------------------------------------------------------------------------------------
# Global numeric table: catches any numeric fields not explicitly listed above
# --------------------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ALL NUMERIC COLUMNS DESCRIBE")
print("=" * 100)

num_df = df.copy()
for c in num_df.columns:
    num_df[c] = pd.to_numeric(num_df[c], errors="ignore")

numeric_detected = num_df.select_dtypes(include="number")
if len(numeric_detected.columns) == 0:
    print("[no numeric columns detected]")
else:
    print(numeric_detected.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]).T.to_string())

# --------------------------------------------------------------------------------------
# Show suspicious / worst rows
# --------------------------------------------------------------------------------------
display_cols = [
    "selected_action",
    "raw_model_action",
    "selected_mode",
    "rerank_reason",
    "selected_progress",
    "min_progress",
    "progress_gap",
    "selected_final_speed",
    "selected_comfort_violation",
    "q_selected",
    "selected_rerank_score",
    "valid_action_count",
    "evidence_count",
    "debug.map_success",
    "debug.map_error",
    "fallback",
    "fallback_reason",
    "debug.fallback_reason",
]
display_cols = [c for c in display_cols if c in df.columns]

print("\n" + "=" * 100)
print("WORST 20 BY selected_progress")
print("=" * 100)
if "selected_progress" in df.columns:
    print(df.sort_values("selected_progress", na_position="last")[display_cols].head(20).to_string(index=True))
else:
    print("[missing selected_progress]")

print("\n" + "=" * 100)
print("WORST 20 BY progress_gap")
print("=" * 100)
if "progress_gap" in df.columns:
    print(df.sort_values("progress_gap", na_position="last")[display_cols].head(20).to_string(index=True))
else:
    print("[missing progress_gap]")

print("\n" + "=" * 100)
print("WORST 20 BY selected_comfort_violation")
print("=" * 100)
if "selected_comfort_violation" in df.columns:
    print(df.sort_values("selected_comfort_violation", ascending=False, na_position="last")[display_cols].head(20).to_string(index=True))
else:
    print("[missing selected_comfort_violation]")

print("\n" + "=" * 100)
print("FALLBACK ROWS")
print("=" * 100)
fallback_mask = pd.Series(False, index=df.index)

for c in ["fallback", "fallback_reason", "debug.fallback_reason"]:
    if c in df.columns:
        if c == "fallback":
            fallback_mask = fallback_mask | (df[c].fillna(False).astype(str).str.lower().isin(["true", "1"]))
        else:
            fallback_mask = fallback_mask | df[c].notna()

fb = df[fallback_mask]
print("fallback rows:", len(fb))
if len(fb):
    print(fb[display_cols].head(50).to_string(index=True))

# --------------------------------------------------------------------------------------
# Optional: save flattened CSV for deeper inspection
# --------------------------------------------------------------------------------------
out_csv = path.with_suffix(".flattened.csv")
df.to_csv(out_csv, index=False)
print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)
print("flattened csv:", out_csv.resolve())