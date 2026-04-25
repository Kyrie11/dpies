# Optimized repair summary

This zip was repaired against `implementation_notes.md`, the supplied gap/bug list, and the follow-up request to use the official nuPlan v1.1 devkit path.

## Implemented repairs from the previous optimized version

- Explicit 9-D ego schema: `[x,y,yaw,vx,vy,ax,ay,yaw_rate,speed]`.
- Rotates ego acceleration into ego frame.
- Candidate action generator no longer uses yaw_rate as speed.
- Adds `agent_history_mask`, `agent_future_mask`, `agent_track_id`, `agent_type`, `ego_global_state`, `ego_to_global` to new caches.
- Fixes `dt` hard-coding in evidence conflict generation and teacher jerk cost.
- Fixes future-agent zero-padding bug in label-side geometry queries by using `agent_future_mask`.
- Adds faster SQLite preprocessing through cached lidar timelines and cached lidar boxes.
- Adds `--require-map`, `--disable-map`, `--skip-existing`, and `--max-samples-total` preprocessing options.
- Map extraction records `map_success` and `map_error`; it no longer fails silently in metadata.
- Adds structured `evidence_metadata_json` for retained evidence units.
- Improves dynamic footprint query with horizon-level footprint overlap sum/max.
- Adds component-wise teacher diagnostics (`teacher_components`, `local_cost_sum`).
- Evidence loss defaults to screened pairs, with `loss.loss_on_all_pairs` for the old behavior.
- Adds extra offline metrics and runtime reporting.

## Newly implemented nuPlan v1.1/devkit integration

- Added `dpies.data.scenario_api.ScenarioAPIExtractor`, which uses official `NuPlanScenario` methods for:
  - `get_route_roadblock_ids()`;
  - `get_traffic_light_status_at_iteration(0)`;
  - `get_future_traffic_light_status_history(...)`;
  - `scenario.map_api`.
- Added `dpies.data.devkit_utils` for robust conversion of official traffic-light records, route ids, EgoState and tracked objects.
- Added direct official DB query fallbacks in `NuPlanSQLite` for route roadblock ids and traffic-light status by lidar token.
- Reworked `NuPlanMapProvider.extract_from_api(...)` to consume official `map_api`, route roadblock ids, current traffic lights and future traffic-light label records.
- HD-map rule extraction now preserves structured geometry for:
  - route corridors;
  - red traffic-light lane connectors;
  - stop lines;
  - crosswalk polygons;
  - drivable-area polygons;
  - lane-boundary polylines;
  - speed-limit rules;
  - route-deviation rules.
- Reworked `GeometryQuery` to use shapely polygon/line operations for:
  - precise drivable-area footprint containment / outside area;
  - lane-boundary crossing;
  - stop-line crossing;
  - red traffic-light crossing using current/future red times in labels;
  - crosswalk footprint intersection;
  - speed-limit violation;
  - route-corridor deviation.
- Added `DPIESNuPlanPlanner`, an official `AbstractPlanner` adapter for closed-loop non-reactive and reactive-IDM evaluation.
- Added Hydra planner configs under `configs/simulation/planner/dpies_planner.yaml` and `configs/nuplan/dpies_planner.yaml`.
- Added closed-loop scripts:
  - `scripts/run_closed_loop_nonreactive.sh`;
  - `scripts/run_closed_loop_reactive_idm.sh`;
  - legacy wrappers `scripts/eval_closed_loop_nonreactive.sh` and `scripts/eval_closed_loop_reactive.sh`.
- Updated README with official-devkit preprocessing and closed-loop execution instructions.

## Remaining caveats

- The code was syntax-validated in this container, but the container does not have the official nuPlan devkit or dataset, so full closed-loop simulation could not be run here.
- The closed-loop adapter falls back to a conservative stop trajectory if online inference fails and `fallback_on_error=true`; disable that flag while debugging to surface exceptions.
- Candidate rollouts still use the repository's lightweight ego-frame action generator. For best closed-loop quality, replace lane-change/merge rollouts with lane-centerline/lane-connector spline rollouts.
- Exact map-rule behavior should be visually validated on your installed map version because local map-object geometry APIs can vary slightly across devkit checkouts.

## Validation performed in this container

- `python -S -m py_compile $(find dpies -name '*.py')` passed on all `dpies/*.py` files.
- Full nuPlan closed-loop simulation was not run because the official devkit/data are not installed in this container.
