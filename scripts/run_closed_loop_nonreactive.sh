#!/usr/bin/env bash
set -euo pipefail

: "${NUPLAN_DEVKIT_ROOT:?Set NUPLAN_DEVKIT_ROOT to your official nuplan-devkit v1.1 checkout}"
: "${NUPLAN_DATA_ROOT:?Set NUPLAN_DATA_ROOT to the nuPlan v1.1 data root}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
: "${CHECKPOINT:?Set CHECKPOINT to a trained DPIES checkpoint path}"

export PYTHONPATH="$(pwd):${NUPLAN_DEVKIT_ROOT}:${PYTHONPATH:-}"
export NUPLAN_DATA_ROOT
export NUPLAN_MAPS_ROOT
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-$(pwd)/nuplan_exp}"

PLANNER_CONFIG_DIR="${NUPLAN_DEVKIT_ROOT}/nuplan/planning/script/config/simulation/planner"
mkdir -p "${PLANNER_CONFIG_DIR}"
cp configs/simulation/planner/dpies_planner.yaml "${PLANNER_CONFIG_DIR}/dpies_planner.yaml"

python "${NUPLAN_DEVKIT_ROOT}/nuplan/planning/script/run_simulation.py" \
  +simulation="${SIMULATION_CONFIG:-closed_loop_nonreactive_agents}" \
  planner=dpies_planner \
  planner.dpies_planner.checkpoint="${CHECKPOINT}" \
  planner.dpies_planner.device="${DEVICE:-cuda}" \
  planner.dpies_planner.budget="${BUDGET:-32}" \
  scenario_builder="${SCENARIO_BUILDER:-nuplan}" \
  scenario_filter="${SCENARIO_FILTER:-one_of_each_scenario_type}" \
  scenario_filter.limit_total_scenarios="${LIMIT_SCENARIOS:-20}" \
  worker="${WORKER:-sequential}" \
  experiment_name="${EXPERIMENT_NAME:-dpies_closed_loop_nonreactive}" \
  group="${GROUP:-dpies}" \
  output_dir="${OUTPUT_DIR:-${NUPLAN_EXP_ROOT}/dpies_closed_loop_nonreactive}" \
  ${EXTRA_NUPLAN_ARGS:-}
