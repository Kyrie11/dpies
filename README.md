# DPIES-nuPlan: Decision-Preserving Interaction Evidence Selection

This repository implements the paper method described in the uploaded Method/Appendix and `implementation_notes.md` for nuPlan DB data and PyTorch.

The implementation follows the four-stage design:

1. finite candidate ego action set;
2. high-recall interaction evidence construction;
3. rival-pair screening and pair-conditioned signed evidence prediction;
4. capped global evidence selection under a budget and max-min pairwise action selection.

The code is intentionally modular so you can replace the heuristic candidate/evidence/teacher pieces with stronger nuPlan-devkit integrations later without changing the model/loss/selection path.

---

## 1. Repository layout

```text
configs/
  data.yaml              # default nuPlan paths and preprocessing settings
  model.yaml             # model dimensions
  train.yaml             # training config
  eval.yaml              # offline evaluation config
dpies/
  actions/               # candidate generation, rollout, filtering, coverage metrics
  common/                # geometry, config, io, torch helpers
  data/                  # direct SQLite nuPlan reader, map wrapper, preprocessing
  evidence/              # evidence builder, geometry query, evidence cost
  teacher/               # teacher evaluator, local costs, oracle/rival/evidence labels
  model/                 # scene/action/evidence/pair encoders and network
  selection/             # capped greedy evidence selection and Q scores
  training/              # dataset, collate, losses, train/validate
  evaluation/            # offline metrics, budget curves, closed-loop adapter skeleton
  tools/                 # toy cache and visualization tools
scripts/
  inspect_db.sh
  build_cache_train.sh
  build_cache_val.sh
  train.sh
  eval_offline.sh
  smoke_test.sh
```

---

## 2. Environment

Create an environment and install dependencies:

```bash
cd dpies_nuplan
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional but recommended for richer HD-map features and closed-loop integration:

```bash
# Install the nuPlan devkit version used by your cluster/project.
# The direct SQLite preprocessing path below works without it, but map extraction
# and closed-loop planner wiring are better with the devkit installed.
```

The provided default paths match your data layout:

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/nuplan/data/cache
export MAP_ROOT=/data0/senzeyu2/dataset/nuplan/maps
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

## 3. Sanity check without nuPlan data

Run this first to verify the package, model, losses, capped selection, and evaluation loop:

```bash
bash scripts/smoke_test.sh
```

Equivalent expanded commands:

```bash
python -m dpies.tools.make_toy_cache --output-dir /tmp/dpies_smoke/train --num-samples 16
python -m dpies.tools.make_toy_cache --output-dir /tmp/dpies_smoke/val --num-samples 8

python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir /tmp/dpies_smoke/train \
  --val-cache-dir /tmp/dpies_smoke/val \
  --output-dir /tmp/dpies_smoke/run \
  --override training.epochs=1 data.batch_size=2 data.num_workers=0 model.hidden_dim=64 model.pair_chunk_size=32

python -m dpies.evaluation.evaluate \
  --checkpoint /tmp/dpies_smoke/run/best.pt \
  --cache-dir /tmp/dpies_smoke/val \
  --output-dir /tmp/dpies_smoke/eval
```

---

## 4. Inspect nuPlan DB schema

Because nuPlan DB releases can differ slightly in column names, first inspect one DB:

```bash
python -m dpies.data.schema_probe \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --limit 1
```

or:

```bash
DATA_ROOT=/data0/senzeyu2/dataset/nuplan/data/cache bash scripts/inspect_db.sh
```

The direct SQLite reader expects the usual nuPlan tables such as `lidar_pc`, `ego_pose`, and `lidar_box`. It uses schema introspection for common token/timestamp/box column variants.

---

## 5. Preprocess validation cache

Build validation cache from `/val`:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --map-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-dir ./cache/val \
  --subdirs val \
  --sample-interval-s 1.0 \
  --history-seconds 2.0 \
  --future-seconds 8.0 \
  --dt 0.5 \
  --max-agents 64 \
  --max-actions 32 \
  --max-evidence-units 128 \
  --continue-on-error
```

or:

```bash
bash scripts/build_cache_val.sh
```

For a very small debug run:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --map-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-dir ./cache/val_debug \
  --subdirs val \
  --max-dbs 1 \
  --max-samples-per-db 100 \
  --sample-interval-s 1.0 \
  --continue-on-error
```

---

## 6. Preprocess training cache

Build training cache from all train DB folders:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --map-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-dir ./cache/train \
  --subdirs train_boston train_singapore train_pittsburgh train_vegas_2 \
  --sample-interval-s 1.0 \
  --history-seconds 2.0 \
  --future-seconds 8.0 \
  --dt 0.5 \
  --max-agents 64 \
  --max-actions 32 \
  --max-evidence-units 128 \
  --continue-on-error
```

or:

```bash
bash scripts/build_cache_train.sh
```

For faster experiments, increase `--sample-interval-s` to `2.0` or limit DBs:

```bash
python -m dpies.data.preprocess_nuplan \
  --data-root /data0/senzeyu2/dataset/nuplan/data/cache \
  --map-root /data0/senzeyu2/dataset/nuplan/maps \
  --output-dir ./cache/train_small \
  --subdirs train_boston train_singapore train_pittsburgh train_vegas_2 \
  --max-dbs 20 \
  --max-samples-per-db 200 \
  --sample-interval-s 2.0 \
  --continue-on-error
```

Each cached `.npz` sample contains:

```text
ego_history, agent_history, map_polylines,
actions, action_meta, action_mask,
evidence_features, evidence_type, evidence_cost, evidence_mask,
geometry_query,
teacher_cost, oracle_action_index,
rival_label,
signed_evidence_label, signed_evidence_mask,
logged_ego_future
```

---

## 7. Visualize one cached sample

```bash
python -m dpies.tools.visualize_sample \
  ./cache/val/sample_000000000.npz \
  --output ./sample_debug.png
```

Use this to check that candidate actions and evidence points are reasonable in ego-centric coordinates.

---

## 8. Train

Default training:

```bash
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir ./cache/train \
  --val-cache-dir ./cache/val \
  --output-dir ./runs/dpies_main
```

or:

```bash
bash scripts/train.sh
```

Useful overrides:

```bash
# Smaller model / lower memory
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir ./cache/train \
  --val-cache-dir ./cache/val \
  --output-dir ./runs/dpies_small \
  --override data.batch_size=2 model.hidden_dim=128 model.pair_chunk_size=32

# Different budget and rival count
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir ./cache/train \
  --val-cache-dir ./cache/val \
  --output-dir ./runs/dpies_B16_M6 \
  --override selection.budget=16 selection.top_m=6

# Resume
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir ./cache/train \
  --val-cache-dir ./cache/val \
  --output-dir ./runs/dpies_main \
  --resume ./runs/dpies_main/last.pt
```

Checkpoints:

```text
runs/dpies_main/last.pt
runs/dpies_main/best.pt
runs/dpies_main/metrics.jsonl
runs/dpies_main/config.json
```

---

## 9. Offline validation / testing

Run budget-curve evaluation on validation cache:

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

Outputs:

```text
runs/dpies_main/eval/metrics.json
runs/dpies_main/eval/metrics.csv
```

Main metrics currently reported:

```text
action_match          # 1[hat_a == a_star]
teacher_regret        # J_T(hat_a) - J_T(a_star)
unresolved_rate       # 1[Q(hat_a) <= 0]
screen_recall_at_m    # recall of teacher rival labels by top-M screening
selected_count        # average retained evidence units
```

---

## 10. Method-to-code map

| Paper component | Code |
|---|---|
| Candidate action set A_t | `dpies/actions/action_generator.py`, `rollout.py` |
| Coverage diagnostics | `dpies/actions/coverage_metrics.py` |
| Evidence units E_t | `dpies/evidence/evidence_builder.py` |
| Explicit GeometryQuery | `dpies/evidence/geometry_query.py` |
| Teacher evaluator J_T | `dpies/teacher/teacher_evaluator.py` |
| Rival labels | `dpies/teacher/labels.py::rival_labels` |
| Signed evidence labels | `dpies/teacher/labels.py::signed_evidence_labels` |
| Scene/action/evidence/pair encoders | `dpies/model/encoders.py`, `network.py` |
| Rival screening loss | `dpies/training/losses.py::screening_loss` |
| Evidence loss | `dpies/training/losses.py::evidence_loss` |
| Capped greedy selection | `dpies/selection/capped_greedy.py` |
| Max-min Q action score | `dpies/selection/capped_greedy.py::compute_q_scores` |
| Action identity and hard-negative loss | `dpies/training/losses.py` |

---

## 11. Important implementation notes

### Direct DB reading

The preprocessing path reads nuPlan DBs directly through SQLite. This avoids forcing a specific nuPlan-devkit version. If your DB schema has different column names, run `schema_probe.py` and patch `dpies/data/nuplan_db.py` in one place.

### HD maps

`dpies/data/map_provider.py` tries to use nuPlan's map API when it is installed. If the devkit is not available, preprocessing still runs with dynamic-agent, conflict, gap, low-TTC, and coarse drivable-boundary evidence. For full paper-quality map-rule evidence, install the devkit and verify that `map_provider.extract(...)` returns non-empty polylines and rule units.

### Teacher labels

Logged future ego and agent tracks are used only during preprocessing for:

```text
teacher_cost
oracle_action_index
rival_label
signed_evidence_label
signed_evidence_mask
```

The model input stores only current/history-derived features and `geometry_query` constructed from current/history evidence.

### Discrete selection during training

The selected evidence indices are treated as fixed during backpropagation. Gradients flow through selected signed evidence values, not through the greedy argmax indices.

### Closed-loop nuPlan evaluation

`dpies/evaluation/closed_loop_planner.py` contains the reusable DPIES planner core and a devkit adapter skeleton. The exact `PlannerInput -> cache-style tensor batch -> InterpolatedTrajectory` conversion is nuPlan-devkit-version-specific, so the offline preprocessing/training/evaluation path is the primary runnable path in this package. Once your cluster's nuPlan devkit version is fixed, implement that conversion inside `compute_planner_trajectory` using the same modules called by `preprocess_nuplan.py`.

---

## 12. Recommended experiment sequence

1. `bash scripts/smoke_test.sh`
2. `bash scripts/inspect_db.sh`
3. Build a small validation cache with `--max-dbs 1 --max-samples-per-db 100`.
4. Visualize 20 random cached samples.
5. Build full validation cache.
6. Build a small train cache and overfit/debug.
7. Build full train cache.
8. Train `runs/dpies_main`.
9. Run offline budget curves.
10. Add stronger HD-map extraction and closed-loop adapter for final nuPlan metrics.

---

## 13. Common failure modes

### No DB files found

Check:

```bash
find /data0/senzeyu2/dataset/nuplan/data/cache -name '*.db' | head
```

Then pass the correct subfolders with `--subdirs`.

### Missing `lidar_pc`, `ego_pose`, or `lidar_box` columns

Run:

```bash
python -m dpies.data.schema_probe --data-root $DATA_ROOT --limit 1
```

Patch column names in `dpies/data/nuplan_db.py` if needed.

### GPU out of memory

Use:

```bash
--override data.batch_size=1 model.hidden_dim=128 model.pair_chunk_size=16
```

The signed evidence head is computed in action-pair chunks. Lower `pair_chunk_size` reduces memory.

### `Q(hat_a) <= 0` often

This is an unresolved retained-interface case, not automatically an action-set coverage failure. Evaluate larger budgets and larger `selection.top_m`:

```bash
--override selection.budget=64 selection.top_m=8
```

### Oracle is too imitation-heavy

Tune `TeacherWeights` in `dpies/teacher/teacher_evaluator.py`, especially `imitation_ade`, `imitation_fde`, collision/proximity, progress, and map-rule weights.
