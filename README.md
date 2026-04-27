# DPIES-nuPlan: Decision-Preserving Interaction Evidence Selection

This repository is an implementation-oriented nuPlan codebase for Decision-Preserving Interaction Evidence Selection (DPIES). It follows `implementation_notes.md` as the implementation specification:

1. finite ego candidate action set;
2. high-recall interaction evidence construction;
3. rival-pair screening and pair-conditioned signed evidence prediction;
4. capped global evidence selection under a budget;
5. max-min pairwise action selection.

## Important implementation status

This version fixes several correctness issues in the original skeleton, improves preprocessing speed, and adds an official nuPlan-devkit v1.1 integration path for route roadblocks, traffic-light status/future labels, exact HD-map rule geometry, and closed-loop simulation. Direct SQLite extraction still reads ego/agents efficiently, while route and traffic-light labels use the official devkit query functions when installed. HD-map extraction can also consume the official Scenario/Planner `map_api` directly for closed-loop inference.

### Main changes in this optimized version

- Ego schema is now explicit: `[x, y, yaw, vx, vy, ax, ay, yaw_rate, speed]`.
- Legacy 8-D ego caches are upgraded by the dataset loader, but rebuilding caches is recommended.
- Acceleration is rotated into the ego frame during preprocessing.
- Action generation uses speed from column 8 or recomputes speed from `vx/vy`; it no longer treats yaw rate as speed.
- `dt` is propagated to conflict evidence and teacher jerk cost; hardcoded `0.5` was removed.
- Agent history/future masks are cached to avoid treating missing future boxes as agents at `(0, 0)`.
- Map extraction failure is exposed through `map_success` and `map_error`; use `--require-map` to skip map-missing samples.
- Direct SQLite reader caches lidar rows, ego poses and lidar boxes, replacing repeated `ORDER BY ABS(timestamp - ?)` queries with in-memory nearest-time lookup.
- Evidence units now keep structured JSON metadata (`evidence_metadata_json`) in addition to fixed tensors.
- Evaluation reports additional diagnostics: top-k action match, normalized regret, Q margin, pairwise ranking accuracy, screening precision, decisive-rival miss rate and evidence prediction metrics.
- New `--use-scenario-api` preprocessing mode extracts route roadblocks and current/future traffic lights through the official nuPlan Scenario API.
- HD-map extraction now creates structured route-corridor, traffic-light connector, drivable-area-union and lane-boundary-union rule units.
- GeometryQuery now uses polygon/line checks for drivable containment, lane-boundary crossing, stop-line/crosswalk intersection and route deviation when structured map geometry is available.
- A nuPlan `AbstractPlanner` adapter is provided for closed-loop non-reactive and reactive-IDM simulation.
- nuPlan v1.1 devkit integration now extracts route roadblock ids and current/future traffic-light states through official devkit queries.
- HD-map extraction now preserves exact rule geometry for drivable-area polygons, lane boundaries, stop lines, crosswalks, speed-limit rules and red traffic-light connectors.
- `DPIESNuPlanPlanner` implements the official `AbstractPlanner` interface for closed-loop non-reactive/reactive-IDM simulation.

---

## 1. Environment

```bash
cd dpies_nuplan
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Required for HD-map extraction and closed-loop integration:

```bash
# Install your official nuPlan devkit v1.1 checkout/environment.
# Example when you maintain a local devkit clone:
#   export NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit
#   pip install -e "$NUPLAN_DEVKIT_ROOT"
#
# Preprocessing can still run without the devkit, but route roadblocks,
# traffic-light labels, exact map-rule geometry and closed-loop simulation
# require the devkit and map files.
```

Optional preprocessing stability/performance settings:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Default path environment variables for this repository's preprocessing scripts:

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/nuplan/data/cache
export MAP_ROOT=/data0/senzeyu2/dataset/nuplan/maps
export SENSOR_ROOT=/data0/senzeyu2/dataset/nuplan/sensor_blobs  # optional
```

Expected DB split layout:

```text
$DATA_ROOT/
  val/
  train_boston/
  train_singapore/
  train_pittsburgh/
  train_vegas_2/
```

---

## 2. Sanity check without nuPlan data

```bash
bash scripts/smoke_test.sh
```

Expanded commands:

```bash
python -m dpies.tools.make_toy_cache --output-dir /tmp/dpies_smoke/train --num-samples 16
python -m dpies.tools.make_toy_cache --output-dir /tmp/dpies_smoke/val --num-samples 8

python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_train \
  --val-cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_val \
  --output-dir /data0/senzeyu2/dataset/nuplan/data/cache/run \
  --override training.epochs=1 data.batch_size=2 data.num_workers=0 model.hidden_dim=64 model.pair_chunk_size=32

python -m dpies.evaluation.evaluate \
  --checkpoint /tmp/dpies_smoke/run/best.pt \
  --cache-dir /tmp/dpies_smoke/val \
  --output-dir /tmp/dpies_smoke/eval
```

---

## 3. Inspect nuPlan DB schema

```bash
python -m dpies.data.schema_probe \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --limit 1
```

or:

```bash
DATA_ROOT=/data0/senzeyu2/dataset/nuplan/data/cache bash scripts/inspect_db.sh
```

---

## 4. Preprocess validation cache

Recommended validation cache command:

```bash
python -m dpies.data.preprocess_nuplan --data-root /data0/senzeyu2/dataset/nuplan/data/cache --map-root /data0/senzeyu2/dataset/nuplan/maps --output-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_val --subdirs val --sample-interval-s 1.0 --history-seconds 2.0 --future-seconds 8.0 --dt 0.5 --max-agents 64 --agent-radius-m 80 --max-actions 32 --max-evidence-units 128 --max-map-polylines 128 --max-map-points 20 --map-radius-m 50 --map-version nuplan-maps-v1.0 --require-map --continue-on-error --skip-existing

```

Script form:

```bash
bash scripts/build_cache_val.sh
```

Strict HD-map mode:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --output-dir ./cache/val_map_required \
  --subdirs val \
  --require-map \
  --continue-on-error
```

Recommended v1.1 official-devkit Scenario API mode:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --sensor-root "${SENSOR_ROOT:-}" \
  --output-dir ./cache/val_scenario_api \
  --subdirs val \
  --use-scenario-api \
  --require-map \
  --sample-interval-s 1.0 \
  --history-seconds 2.0 \
  --future-seconds 8.0 \
  --dt 0.5 \
  --continue-on-error
```

`--use-scenario-api` uses `NuPlanScenario.get_route_roadblock_ids()`, `get_traffic_light_status_at_iteration()`, `get_future_traffic_light_status_history()` and `scenario.map_api`.  This is the recommended cache-building path for full v1.1 DB experiments.

If your DB copy is writable, optional SQLite indexes can help first-run preprocessing:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --output-dir ./cache/val \
  --subdirs val \
  --create-sqlite-indexes \
  --continue-on-error
```

Script toggles:

```bash
REQUIRE_MAP=1 bash scripts/build_cache_val.sh
CREATE_SQLITE_INDEXES=1 bash scripts/build_cache_val.sh
SAMPLE_INTERVAL_S=2.0 bash scripts/build_cache_val.sh
```

---

## 5. Preprocess training cache

```bash
python -m dpies.data.preprocess_nuplan --data-root /data0/senzeyu2/dataset/nuplan/data/cache --map-root /data0/senzeyu2/dataset/nuplan/maps --output-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_train --subdirs train_boston train_pittsburgh train_vegas_2 train_singapore --sample-interval-s 1.0 --history-seconds 2.0 --future-seconds 8.0 --dt 0.5 --max-agents 64 --agent-radius-m 80 --max-actions 32 --max-evidence-units 128 --max-map-polylines 128 --max-map-points 20 --map-radius-m 50 --map-version nuplan-maps-v1.0 --require-map --continue-on-error --skip-existing

```

Script form:

```bash
bash scripts/build_cache_train.sh
```

Faster experimental preprocessing options:

```bash
SAMPLE_INTERVAL_S=2.0 bash scripts/build_cache_train.sh

python -m dpies.data.preprocess_nuplan \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --output-dir ./cache/train_small \
  --subdirs train_boston train_singapore train_pittsburgh train_vegas_2 \
  --max-dbs 20 \
  --max-samples-per-db 200 \
  --sample-interval-s 2.0 \
  --continue-on-error
```

### Cache fields

Each `.npz` sample contains the original training tensors plus new schema/debug fields:

```text
ego_history                         # [H,9], x,y,yaw,vx,vy,ax,ay,yaw_rate,speed
agent_history, agent_history_mask
agent_future, agent_future_mask     # labels/debug only; not a model input
agent_mask, agent_track_id, agent_type
map_polylines, map_masks
actions, action_meta, action_mask
evidence_features, evidence_type, evidence_cost, evidence_mask
geometry_query
teacher_cost, teacher_components, local_cost_sum
oracle_action_index, rival_label
signed_evidence_label, signed_evidence_mask
logged_ego_future
ego_to_global                       # x,y,yaw at planning time
metadata_json, evidence_metadata_json, route_info_json, traffic_lights_json
```

Rebuild old caches after this update if possible. The dataset loader can upgrade legacy 8-D `ego_history` to 9-D, but old caches do not contain future masks, agent ids or structured evidence metadata.

---

## 6. Visualize one cached sample

```bash
python -m dpies.tools.visualize_sample \
  ./cache/val/sample_000000000.npz \
  --output ./sample_debug.png
```

---

## 7. Train

```bash
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_train \
  --val-cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_val \
  --output-dir ./runs/dpies_main
```

or:

```bash
bash scripts/train.sh
```

---

## 8. Offline evaluation

```bash
python -m dpies.evaluation.evaluate \
  --config configs/eval.yaml \
  --checkpoint ./runs/dpies_main/best.pt \
  --cache-dir ./cache/val \
  --output-dir ./runs/dpies_main/eval
```

or:

```bash
bash scripts/eval_offline.sh
```

The evaluator now writes `metrics.json` and `metrics.csv` with budget curves and diagnostics, including action match, top-k match, teacher regret, normalized regret, unresolved rate, Q margin, pairwise ranking accuracy, screening recall/precision, decisive-rival miss rate and evidence prediction metrics.

---

## 9. Official nuPlan closed-loop evaluation

This repository provides a Hydra-instantiable official nuPlan planner:

```text
dpies.evaluation.closed_loop_planner.DPIESNuPlanPlanner
```

The planner implements the official `AbstractPlanner` flow. During initialization it receives `PlannerInitialization.route_roadblock_ids`, `mission_goal`, and `map_api`. At every simulation step it reads ego history, tracked-object observations, and `PlannerInput.traffic_light_data`; builds candidate actions, interaction evidence units, exact map-aware `GeometryQuery` tensors; runs the trained DPIES model; performs capped evidence selection and max-min action selection; and returns the selected rollout as a global `InterpolatedTrajectory`.

### 9.1 Planner config

The included config is:

```text
configs/simulation/planner/dpies_planner.yaml
```

The helper scripts copy it into the official devkit planner config directory:

```text
$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/config/simulation/planner/dpies_planner.yaml
```

so the official `run_simulation.py` can be launched with `planner=dpies_planner`.

### 9.2 Environment variables

```bash
export NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit          # official v1.1 devkit checkout
export NUPLAN_DATA_ROOT=/path/to/nuplan-v1.1              # full DB set root
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps              # HD map root
export NUPLAN_EXP_ROOT=/path/to/nuplan/experiments        # simulation outputs
export CHECKPOINT=$PWD/runs/dpies_main/best.pt
export PYTHONPATH="$PWD:$NUPLAN_DEVKIT_ROOT:${PYTHONPATH:-}"
```

Run a tiny job first:

```bash
LIMIT_SCENARIOS=1 bash scripts/run_closed_loop_nonreactive.sh
```

### 9.3 Closed-loop non-reactive agents

```bash
CHECKPOINT=$PWD/runs/dpies_main/best.pt \
NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit \
NUPLAN_DATA_ROOT=/path/to/nuplan-v1.1 \
NUPLAN_MAPS_ROOT=/path/to/nuplan/maps \
LIMIT_SCENARIOS=20 \
bash scripts/run_closed_loop_nonreactive.sh
```

The script calls the official devkit simulation entry point with:

```bash
+simulation=closed_loop_nonreactive_agents
planner=dpies_planner
planner.dpies_planner.checkpoint=$CHECKPOINT
```

### 9.4 Closed-loop reactive-IDM agents

```bash
CHECKPOINT=$PWD/runs/dpies_main/best.pt \
NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit \
NUPLAN_DATA_ROOT=/path/to/nuplan-v1.1 \
NUPLAN_MAPS_ROOT=/path/to/nuplan/maps \
LIMIT_SCENARIOS=20 \
bash scripts/run_closed_loop_reactive_idm.sh
```

The script calls:

```bash
+simulation=closed_loop_reactive_agents
planner=dpies_planner
planner.dpies_planner.checkpoint=$CHECKPOINT
```

### 9.5 Useful overrides

```bash
# CPU debugging
DEVICE=cpu LIMIT_SCENARIOS=1 bash scripts/run_closed_loop_nonreactive.sh

# Change evidence budget
BUDGET=16 LIMIT_SCENARIOS=20 bash scripts/run_closed_loop_reactive_idm.sh

# Pass arbitrary official devkit/Hydra overrides
EXTRA_NUPLAN_ARGS="scenario_filter.scenario_types='[nearby_dense_vehicle_traffic]'" \
  LIMIT_SCENARIOS=10 bash scripts/run_closed_loop_reactive_idm.sh
```

The legacy script names remain available:

```bash
DPIES_CHECKPOINT=$PWD/runs/dpies_main/best.pt LIMIT_TOTAL_SCENARIOS=20 bash scripts/eval_closed_loop_nonreactive.sh
DPIES_CHECKPOINT=$PWD/runs/dpies_main/best.pt LIMIT_TOTAL_SCENARIOS=20 bash scripts/eval_closed_loop_reactive.sh
```

### 9.6 Current closed-loop implementation notes

- The planner does **not** use logged future tracks or future traffic lights online. Current traffic-light data comes from `PlannerInput.traffic_light_data`; future traffic lights are only used in offline cache label construction.
- The planner queries route roadblocks from `PlannerInitialization.route_roadblock_ids` and exact HD-map objects from `PlannerInitialization.map_api`.
- Drivable-area containment, lane-boundary crossing, stop-line/crosswalk intersection, speed-limit violation, and route-deviation checks use structured map polygons/lines when available.
- If online inference fails and `fallback_on_error=true`, the planner returns a conservative stop rollout instead of crashing the whole simulation. Set `planner.dpies_planner.fallback_on_error=false` when debugging.
- Candidate action generation still uses the repository's lightweight rollout generator; for highest closed-loop quality, the next major improvement should be fully lane-centerline/lane-connector spline rollouts.

## 10. Development status and important caveats

Implemented in this version:

- Official Scenario API preprocessing path for route roadblocks, current traffic lights, future traffic-light labels, and `scenario.map_api`.
- Direct official DB query fallback for route roadblocks and current/future traffic-light status when available.
- Structured HD-map rule extraction for route corridor, red traffic-light connectors, stop lines, crosswalks, drivable area, lane boundary, speed limit, and route deviation.
- Exact map-aware `GeometryQuery` when structured geometry is present.
- Official nuPlan `AbstractPlanner` adapter for closed-loop non-reactive and reactive-IDM simulation.

Still recommended before large experiments:

- Visually inspect at least 50 cached samples and 10 closed-loop scenarios using nuBoard / sample visualizations.
- Verify `map_success`, `num_route_roadblocks`, traffic-light counts, and rule-unit counts in the preprocessing manifest.
- Tune teacher weights after exact map-rule costs are active.
- Replace lightweight ego-frame lane-change/merge rollouts with lane-centerline/lane-connector rollouts for stronger closed-loop performance.


## 11. Test generated npz file 

```bash
python scripts/check_npz_cache.py   --cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/processed_train   --limit 1000   --show-files 5
```