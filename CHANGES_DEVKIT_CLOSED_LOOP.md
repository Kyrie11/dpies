# Devkit v1.1 map-data and closed-loop integration changes

This update assumes nuPlan v1.1, the official devkit, and a full DB set.

## Implemented

- Added official Scenario/devkit extraction helpers:
  - route roadblock ids via `NuPlanScenario.get_route_roadblock_ids()` or devkit DB query helpers;
  - current traffic-light state via `get_traffic_light_status_at_iteration()` / planner input traffic-light data;
  - future traffic-light histories for offline teacher-label construction only.
- Extended HD-map extraction:
  - lane/lane-connector/roadblock/crosswalk/stop-line/drivable-area/lane-boundary/traffic-light connector layers;
  - exact rule geometry stored in `evidence_metadata_json`;
  - route corridor metadata and route-deviation rule units;
  - red traffic-light connector rule units.
- Extended `GeometryQuery`:
  - stop-line and red-light crossing from line/polygon intersection;
  - crosswalk intersection;
  - drivable-area footprint containment / outside-area ratio;
  - lane-boundary crossing;
  - route-deviation distance;
  - speed-limit violation from map rule metadata.
- Added a nuPlan `AbstractPlanner` adapter:
  - `dpies.evaluation.closed_loop_planner.DPIESNuPlanPlanner`;
  - online DPIES batch construction from `PlannerInput` and `PlannerInitialization`;
  - model inference, capped evidence selection, max-min action selection;
  - selected ego-centric candidate converted to `InterpolatedTrajectory`;
  - conservative stop fallback on runtime error.
- Added closed-loop helper scripts:
  - `scripts/run_closed_loop_nonreactive.sh`;
  - `scripts/run_closed_loop_reactive_idm.sh`;
  - backward-compatible wrappers `scripts/eval_closed_loop_nonreactive.sh` and `scripts/eval_closed_loop_reactive.sh`.
- Updated README closed-loop instructions.

## Validation performed here

- Syntax validation with `python3 -S -m py_compile` over the package.
- Full nuPlan closed-loop execution was not run in this container because the official devkit, maps, and DB set are not mounted here.

## Remaining engineering notes

- Candidate rollouts still use the repository's ego-frame fallback action library. The generator now accepts map/route/traffic context, but high-quality lane-centerline/lane-connector spline rollout remains a recommended next improvement.
- Hydra simulation names can differ slightly across devkit checkouts. The scripts use common v1.1 names and document where to adjust them.
- The fallback trajectory is conservative stop; replace it with IDM-style fallback if desired.
