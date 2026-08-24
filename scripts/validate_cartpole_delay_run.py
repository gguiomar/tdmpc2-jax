#!/usr/bin/env python3
"""Validate one terminal Cartpole-delay run before the steward accepts it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ANCHORS = (0, 100_000, 150_000, 250_000, 350_000, 450_000, 500_000)
HORIZONS = tuple(range(2, 9))
PROBE_COUNTS = (0, 2, 4, 8, 16, 32, 64)
ROUGHNESS_COUNTS = PROBE_COUNTS[1:]
RETURN_COUNTS = (8, 16, 32, 64, 128)


def contract_expectations(contract: str, min_steps: int) -> dict[str, object]:
  """Return launcher settings that distinguish full, smoke, and scout runs."""
  if contract == "auto":
    contract = "smoke" if int(min_steps) < 500_000 else "full"
  if contract not in {"full", "smoke", "stabilization"}:
    raise ValueError(f"unsupported validation contract {contract!r}")
  if contract == "smoke":
    return {
        "name": contract,
        "save_interval_steps": 24_000,
        "calibration_anchors": [20_000],
        "timing_warmup_calls": 1,
        "timing_repetitions": 2,
        "reference_eval_steps": 32,
        "conditional_reference_eval_steps": 32,
        "action_delay_schedule_enabled": True,
        "evaluation_interval_steps": 24_000,
        "evaluation_num_episodes": 4,
    }
  if contract == "stabilization":
    return {
        "name": contract,
        "save_interval_steps": 50_000,
        "calibration_anchors": [100_000, 250_000, 450_000],
        "timing_warmup_calls": 5,
        "timing_repetitions": 30,
        "reference_eval_steps": 256,
        "conditional_reference_eval_steps": 500,
        "action_delay_schedule_enabled": False,
        "evaluation_interval_steps": 2_500,
        "evaluation_num_episodes": 10,
    }
  return {
      "name": contract,
      "save_interval_steps": 50_000,
      "calibration_anchors": [100_000, 250_000, 450_000],
      "timing_warmup_calls": 5,
      "timing_repetitions": 30,
      "reference_eval_steps": 256,
      "conditional_reference_eval_steps": 500,
      "action_delay_schedule_enabled": True,
      "evaluation_interval_steps": 50_000,
      "evaluation_num_episodes": 20,
  }


def nonempty(path: Path) -> bool:
  return path.is_file() and path.stat().st_size > 0


def finite_csv(path: Path, numeric_columns: tuple[str, ...]) -> tuple[bool, int, str]:
  if not nonempty(path):
    return False, 0, f"missing or empty {path}"
  rows = 0
  try:
    with path.open(newline="") as handle:
      reader = csv.DictReader(handle)
      fields = set(reader.fieldnames or [])
      missing = set(numeric_columns) - fields
      if missing:
        return False, 0, f"{path} missing columns {sorted(missing)}"
      for row in reader:
        rows += 1
        for column in numeric_columns:
          value = float(row[column])
          if not math.isfinite(value):
            return False, rows, f"non-finite {column} at row {rows} in {path}"
  except Exception as exc:
    return False, rows, f"cannot parse {path}: {exc}"
  return rows > 0, rows, ""


def csv_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return list(csv.DictReader(handle))


def csv_integral(value: str) -> int:
  """Parse an integer-valued CSV field, including pandas-style ``0.0``."""
  numeric = float(value)
  if not math.isfinite(numeric) or not numeric.is_integer():
    raise ValueError(f"expected an integer-valued CSV field, got {value!r}")
  return int(numeric)


def require_grid(errors: list[str], *,
                 label: str,
                 actual: set[tuple],
                 expected: set[tuple]) -> None:
  missing = sorted(expected - actual)
  extra = sorted(actual - expected)
  if missing:
    errors.append(f"{label} missing {len(missing)} cells; first={missing[:5]}")
  if extra:
    errors.append(f"{label} has {len(extra)} unexpected cells; first={extra[:5]}")


def validate_resolved_config(config_path: Path,
                             *,
                             controller: str,
                             score_mode: str | None,
                             seed: int | None,
                             min_steps: int,
                             anchors: tuple[int, ...],
                             require_probe_timing: bool,
                             require_reference_probe: bool,
                             contract: str = "auto") -> list[str]:
  """Checks the resolved Hydra config against the frozen pilot contract."""
  errors: list[str] = []
  if not nonempty(config_path):
    return [f"missing resolved Hydra config {config_path}"]
  try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
  except Exception as exc:
    return [f"cannot load resolved Hydra config {config_path}: {exc}"]

  contract_values = contract_expectations(contract, min_steps)
  calibration_anchors = contract_values["calibration_anchors"]
  expected: dict[str, object] = {
      "controller": controller,
      "score_mode": score_mode,
      "seed": seed,
      "max_steps": int(min_steps),
      "save_interval_steps": contract_values["save_interval_steps"],
      "checkpoint_buffer": True,
      "update_chunk_size": 128,
      "collect_chunk_steps": 100,
      "seed_steps_override": 4_000,
      "artifact_capture_enabled": True,
      "artifact_anchor_enabled": True,
      "artifact_anchor_steps": list(anchors),
      "probe_timing.enabled": require_probe_timing,
      "probe_timing.anchor_steps": calibration_anchors,
      "probe_timing.probe_counts": list(PROBE_COUNTS),
      "probe_timing.warmup_calls": contract_values["timing_warmup_calls"],
      "probe_timing.repetitions": contract_values["timing_repetitions"],
      "reference_probe.enabled": require_reference_probe,
      "reference_probe.anchor_steps": calibration_anchors,
      "reference_probe.num_env_eval_replicas": 128,
      "reference_probe.env_eval_steps": contract_values["reference_eval_steps"],
      "reference_probe.conditional_reference_env_eval_steps": (
          contract_values["conditional_reference_eval_steps"]
      ),
      "env.backend": "mjx_dmc",
      "env.env_id": "cartpole-swingup",
      "env.num_envs": 8,
      "env.utd_ratio": 1.0,
      "env.asynchronous": False,
      "env.mjx_dmc.task": "cartpole-swingup",
      "env.mjx_dmc.action_repeat": 2,
      "env.mjx_dmc.episode_length": 500,
      "env.mjx_dmc.enable_domain_randomization": False,
      "env.mjx_dmc.enable_observation_noise": False,
      "env.mjx_dmc.observation_noise_scale": 0.0,
      "env.mjx_dmc.base_action_delay": 0,
      "env.mjx_dmc.action_delay_schedule_enabled": (
          contract_values["action_delay_schedule_enabled"]
      ),
      "env.mjx_dmc.action_delay_observation_enabled": True,
      "env.mjx_dmc.reset_pool_size": 64,
      "dense_rhs.enabled": controller == "adaptive",
      "dense_rhs.initial_horizon": 8 if controller == "adaptive" else 3,
      "dense_rhs.start_query_step": 20_000,
      "dense_rhs.query_interval_steps": 20_000,
      "dense_rhs.horizons": list(HORIZONS),
      "dense_rhs.hmax": 8,
      "dense_rhs.num_roughness_probes": 64,
      "dense_rhs.score_mode": score_mode,
      "dense_rhs.decision_rule": "paired_lcb",
      "dense_rhs.confidence_z": 1.6448536,
      "dense_rhs.switch_threshold": 0.02,
      "dense_rhs.phase_pruning_enabled": False,
      "dense_rhs.local_window_radius": 1,
      "dense_rhs.max_transition_delta": 1,
      "dense_rhs.num_env_eval_replicas": 32,
      "dense_rhs.env_eval_steps": 256,
      "dense_rhs.query_population_size": 256,
      "dense_rhs.query_policy_prior_samples": 12,
      "dense_rhs.query_num_elites": 32,
      "dense_rhs.query_mppi_iterations": 4,
      "dense_rhs.query_temperature": 0.5,
      "dense_rhs.candidate_budget": {
          "A": 3, "B1": 3, "B2": 3, "B3": 3, "B4": 3,
      },
      "scripted_horizon.enabled": controller == "scripted",
      "scripted_horizon.schedule_steps": [0, 150_000, 350_000],
      "scripted_horizon.schedule_values": [3, 7, 3],
      "evaluation.enabled": True,
      "evaluation.interval_steps": contract_values["evaluation_interval_steps"],
      "evaluation.num_episodes": contract_values["evaluation_num_episodes"],
      "evaluation.clean": True,
      "tdmpc2.horizon": 8 if controller == "adaptive" else 3,
      "tdmpc2.population_size": 512,
      "tdmpc2.policy_prior_samples": 24,
      "tdmpc2.num_elites": 64,
      "tdmpc2.mppi_iterations": 6,
      "tdmpc2.temperature": 0.5,
      "wandb.enabled": False,
  }
  for path, wanted in expected.items():
    if wanted is None:
      continue
    actual = OmegaConf.select(cfg, path)
    if OmegaConf.is_config(actual):
      actual = OmegaConf.to_container(actual, resolve=True)
    if actual != wanted:
      errors.append(f"resolved config {path}={actual!r}, expected {wanted!r}")
  return errors


def validate_adaptive_coverage(run_dir: Path, args) -> list[str]:
  """Requires the full deployed/reference calibration design, not nonempty files."""
  errors: list[str] = []
  expected_query_steps = set(range(20_000, int(args.min_steps), 20_000))
  reference_steps = (
      {20_000}
      if int(args.min_steps) < 500_000
      else {100_000, 250_000, 450_000}
  )
  try:
    query_rows = csv_rows(run_dir / "metrics/horizon_queries.csv")
    query_steps = {int(row["step"]) for row in query_rows}
    if query_steps != expected_query_steps:
      errors.append(
          "online query coverage mismatch: "
          f"actual={sorted(query_steps)}, expected={sorted(expected_query_steps)}"
      )
  except Exception as exc:
    errors.append(f"cannot validate online query coverage: {exc}")
    query_steps = expected_query_steps

  try:
    roughness_rows = csv_rows(run_dir / "metrics/roughness_all_horizons.csv")
    actual = {
        (
            row["source"], int(row["step"]), int(row["horizon"]),
            int(row["probe_count"]),
        )
        for row in roughness_rows
    }
    expected = {
        ("deployed_query", step, horizon, count)
        for step in expected_query_steps
        for horizon in HORIZONS
        for count in ROUGHNESS_COUNTS
    }
    if args.require_reference_probe:
      expected |= {
          ("reference_query", step, horizon, count)
          for step in reference_steps
          for horizon in HORIZONS
          for count in ROUGHNESS_COUNTS
      }
    require_grid(
        errors,
        label="all-horizon roughness grid",
        actual=actual,
        expected=expected,
    )
  except Exception as exc:
    errors.append(f"cannot validate all-horizon roughness grid: {exc}")

  try:
    bootstrap_rows = csv_rows(run_dir / "metrics/roughness_bootstrap.csv")
    actual = {
        (
            row["source"], int(row["step"]),
            int(row["candidate_horizon"]), int(row["probe_count"]),
        )
        for row in bootstrap_rows
    }
    expected = {
        ("deployed_query", step, horizon, count)
        for step in expected_query_steps
        for horizon in HORIZONS
        for count in ROUGHNESS_COUNTS
    }
    if args.require_reference_probe:
      expected |= {
          ("reference_query", step, horizon, count)
          for step in reference_steps
          for horizon in HORIZONS
          for count in ROUGHNESS_COUNTS
      }
    require_grid(
        errors,
        label="roughness bootstrap grid",
        actual=actual,
        expected=expected,
    )
  except Exception as exc:
    errors.append(f"cannot validate roughness bootstrap grid: {exc}")

  if args.require_probe_timing:
    try:
      timing_rows = csv_rows(run_dir / "metrics/probe_timing.csv")
      require_grid(
          errors,
          label="probe timing grid",
          actual={
              (csv_integral(row["step"]), csv_integral(row["probe_count"]))
              for row in timing_rows
          },
          expected={
              (step, count) for step in reference_steps for count in PROBE_COUNTS
          },
      )
    except Exception as exc:
      errors.append(f"cannot validate probe timing grid: {exc}")

  try:
    decision_rows = csv_rows(
        run_dir / "metrics/decision_probe_calibration.csv"
    )
    actual = {
        (row["source"], int(row["step"]), int(row["probe_count"]))
        for row in decision_rows
    }
    expected = {
        ("deployed_query", step, count)
        for step in expected_query_steps
        for count in ROUGHNESS_COUNTS
    }
    if args.require_reference_probe:
      expected |= {
          ("reference_query", step, count)
          for step in reference_steps
          for count in ROUGHNESS_COUNTS
      }
    require_grid(
        errors,
        label="nested decision grid",
        actual=actual,
        expected=expected,
    )
  except Exception as exc:
    errors.append(f"cannot validate nested decision grid: {exc}")

  if args.require_reference_probe:
    grids = (
        (
            "return calibration grid",
            run_dir / "metrics/return_probe_calibration.csv",
            lambda row: (
                int(row["step"]), int(row["candidate_horizon"]),
                int(row["return_replicas"]),
            ),
            {
                (step, horizon, count)
                for step in reference_steps
                for horizon in HORIZONS
                for count in RETURN_COUNTS
            },
        ),
        (
            "conditional return reference grid",
            run_dir / "metrics/conditional_return_reference.csv",
            lambda row: (int(row["step"]), int(row["candidate_horizon"])),
            {
                (step, horizon)
                for step in reference_steps
                for horizon in HORIZONS
            },
        ),
        (
            "return decision calibration grid",
            run_dir / "metrics/return_decision_calibration.csv",
            lambda row: (int(row["step"]), int(row["return_replicas"])),
            {
                (step, count)
                for step in reference_steps
                for count in RETURN_COUNTS
            },
        ),
    )
    for label, path, key_fn, expected in grids:
      try:
        require_grid(
            errors,
            label=label,
            actual={key_fn(row) for row in csv_rows(path)},
            expected=expected,
        )
      except Exception as exc:
        errors.append(f"cannot validate {label}: {exc}")

  projection_path = run_dir / "metrics/roughness_projections.npz"
  if nonempty(projection_path):
    try:
      with np.load(projection_path, allow_pickle=False) as projections:
        actual = {
            (
                "reference" if bool(reference) else "deployed",
                int(step), int(horizon), int(direction),
            )
            for step, horizon, direction, reference in zip(
                projections["step"], projections["horizon"],
                projections["direction"], projections["is_reference"],
            )
        }
      expected = {
          ("deployed", step, horizon, direction)
          for step in expected_query_steps
          for horizon in HORIZONS
          for direction in range(64)
      }
      if args.require_reference_probe:
        expected |= {
            ("reference", step, horizon, direction)
            for step in reference_steps
            for horizon in HORIZONS
            for direction in range(64)
        }
      require_grid(
          errors,
          label="roughness projection grid",
          actual=actual,
          expected=expected,
      )
    except Exception as exc:
      errors.append(f"cannot validate roughness projection grid: {exc}")

  paired_path = run_dir / "metrics/paired_returns.npz"
  if nonempty(paired_path):
    try:
      with np.load(paired_path, allow_pickle=False) as paired:
        source = np.asarray(paired["source"]).astype(str)
        steps = np.asarray(paired["step"], dtype=np.int64)
        horizons = np.asarray(paired["candidate_horizon"], dtype=np.int32)
        replicas = np.asarray(paired["replica"], dtype=np.int32)
      valid_sources = {"deployed", "reference", "conditional_reference"}
      unexpected_sources = set(source.tolist()) - valid_sources
      if unexpected_sources:
        errors.append(
            f"paired returns contain invalid sources {sorted(unexpected_sources)}"
        )
      source_cardinality = {"deployed": 32}
      if args.require_reference_probe:
        source_cardinality.update(
            reference=128,
            conditional_reference=32,
        )
      for current_source, expected_k in source_cardinality.items():
        source_mask = source == current_source
        groups = {
            (int(step), int(horizon))
            for step, horizon in zip(steps[source_mask], horizons[source_mask])
        }
        if current_source == "deployed":
          if {step for step, _ in groups} != expected_query_steps:
            errors.append(
                "deployed paired-return steps do not match online query coverage"
            )
        else:
          require_grid(
              errors,
              label=f"{current_source} paired-return groups",
              actual=groups,
              expected={
                  (step, horizon)
                  for step in reference_steps
                  for horizon in HORIZONS
              },
          )
        for step, horizon in groups:
          group_mask = source_mask & (steps == step) & (horizons == horizon)
          actual_replicas = set(replicas[group_mask].tolist())
          if actual_replicas != set(range(expected_k)):
            errors.append(
                f"{current_source} step={step} h={horizon} has "
                f"{len(actual_replicas)} replica ids; expected exactly {expected_k}"
            )
    except Exception as exc:
      errors.append(f"cannot validate paired-return coverage: {exc}")
  return errors


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument(
      "--controller", choices=("adaptive", "scripted", "fixed"), required=True
  )
  parser.add_argument("--min-steps", type=int, default=500_000)
  parser.add_argument(
      "--anchor-steps",
      default=",".join(str(step) for step in ANCHORS),
  )
  parser.add_argument("--require-probe-timing", action="store_true")
  parser.add_argument("--require-reference-probe", action="store_true")
  parser.add_argument("--expected-run-id")
  parser.add_argument("--expected-score-mode")
  parser.add_argument("--expected-seed", type=int)
  parser.add_argument("--expected-commit")
  parser.add_argument("--expected-config-hash")
  parser.add_argument(
      "--contract",
      choices=("auto", "full", "smoke", "stabilization"),
      default="auto",
  )
  args = parser.parse_args()
  run_dir = args.run_dir.resolve()
  anchors = tuple(sorted({
      int(value.strip())
      for value in args.anchor_steps.split(",")
      if value.strip()
  }))
  errors: list[str] = []
  report: dict[str, object] = {
      "run_dir": str(run_dir),
      "controller": args.controller,
  }

  manifest_path = run_dir / "artifacts/run_manifest.json"
  if not nonempty(manifest_path):
    errors.append(f"missing manifest {manifest_path}")
    manifest: dict[str, object] = {}
  else:
    try:
      manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
      errors.append(f"invalid manifest: {exc}")
      manifest = {}
  report["manifest"] = manifest
  try:
    final_step = int(manifest.get("final_step", -1))
  except (TypeError, ValueError):
    final_step = -1
  if final_step < args.min_steps:
    errors.append(f"final_step={final_step} is below {args.min_steps}")
  if manifest.get("status") not in {"completed", "valid"}:
    errors.append(f"manifest status is {manifest.get('status')!r}, not completed")
  expected_manifest = {
      "run_id": args.expected_run_id,
      "controller": args.controller,
      "score_mode": args.expected_score_mode,
      "seed": args.expected_seed,
      "git_commit": args.expected_commit,
      "profile_config_hash": args.expected_config_hash,
  }
  for field, expected in expected_manifest.items():
    if expected is not None and manifest.get(field) != expected:
      errors.append(
          f"manifest {field}={manifest.get(field)!r}, expected {expected!r}"
      )
  resolved_config_errors = validate_resolved_config(
      run_dir / ".hydra/config.yaml",
      controller=args.controller,
      score_mode=args.expected_score_mode,
      seed=args.expected_seed,
      min_steps=args.min_steps,
      anchors=anchors,
      require_probe_timing=args.require_probe_timing,
      require_reference_probe=args.require_reference_probe,
      contract=args.contract,
  )
  errors.extend(resolved_config_errors)
  report["resolved_config_errors"] = resolved_config_errors
  config_path = run_dir / ".hydra/config.yaml"
  if nonempty(config_path):
    actual_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if manifest.get("config_sha256") != actual_config_sha256:
      errors.append(
          "manifest config_sha256 does not match the resolved Hydra config"
      )

  checks = {
      "scalars": finite_csv(
          run_dir / "metrics/scalars.csv", ("step", "value")
      ),
      "episodes": finite_csv(
          run_dir / "metrics/episodes.csv",
          ("step", "episode_return", "episode_length"),
      ),
      "trajectory_summary": finite_csv(
          run_dir / "metrics/trajectory_summary.csv",
          (
              "step", "effective_delay", "episode_return",
              "upright_fraction", "mean_abs_command_applied_mismatch",
          ),
      ),
  }
  if args.controller == "adaptive":
    checks.update(
        {
            "horizon_queries": finite_csv(
                run_dir / "metrics/horizon_queries.csv",
                ("step", "selected_horizon", "robust_return_best"),
            ),
            "probe_calibration": finite_csv(
                run_dir / "metrics/probe_calibration.csv",
                ("step", "probe_count", "roughness"),
            ),
            "decision_probe_calibration": finite_csv(
                run_dir / "metrics/decision_probe_calibration.csv",
                ("step", "probe_count", "selected_horizon", "switch"),
            ),
            "roughness_all_horizons": finite_csv(
                run_dir / "metrics/roughness_all_horizons.csv",
                ("step", "probe_count", "roughness"),
            ),
            "roughness_bootstrap": finite_csv(
                run_dir / "metrics/roughness_bootstrap.csv",
                (
                    "step", "probe_count", "roughness", "bootstrap_se",
                    "bootstrap_lcb95", "bootstrap_ucb95",
                ),
            ),
        }
    )
    if args.require_probe_timing:
      checks["probe_timing"] = finite_csv(
          run_dir / "metrics/probe_timing.csv",
          ("step", "probe_count", "wall_time_s"),
      )
    if args.require_reference_probe:
      checks["return_probe_calibration"] = finite_csv(
          run_dir / "metrics/return_probe_calibration.csv",
          ("step", "return_replicas", "return_mean"),
      )
      checks["return_decision_calibration"] = finite_csv(
          run_dir / "metrics/return_decision_calibration.csv",
          ("step", "return_replicas", "selected_horizon", "switch"),
      )
      checks["conditional_return_reference"] = finite_csv(
          run_dir / "metrics/conditional_return_reference.csv",
          (
              "step", "candidate_horizon", "return_mean",
              "best_by_sample_mean_horizon", "paired_gap_to_best_se",
              "probability_replica_best",
          ),
      )
    roughness_projection_path = run_dir / "metrics/roughness_projections.npz"
    if not nonempty(roughness_projection_path):
      errors.append(f"missing roughness projections {roughness_projection_path}")
    else:
      try:
        with np.load(roughness_projection_path, allow_pickle=False) as projections:
          required = {
              "step", "horizon", "direction", "projection", "is_reference",
          }
          missing = required - set(projections.files)
          if missing:
            errors.append(f"roughness projections missing arrays {sorted(missing)}")
          elif not np.all(np.isfinite(np.asarray(projections["projection"]))):
            errors.append("roughness projections contain non-finite values")
      except Exception as exc:
        errors.append(f"cannot validate roughness projections: {exc}")
    paired_path = run_dir / "metrics/paired_returns.npz"
    if not nonempty(paired_path):
      errors.append(f"missing paired returns {paired_path}")
    else:
      try:
        with np.load(paired_path, allow_pickle=False) as paired:
          required = {
              "step", "candidate_horizon", "replica", "episode_return",
              "source", "is_reference", "is_conditional_reference",
          }
          missing = required - set(paired.files)
          if missing:
            errors.append(f"paired returns missing arrays {sorted(missing)}")
          else:
            lengths = {np.asarray(paired[name]).shape[0] for name in required}
            if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
              errors.append("paired return arrays have inconsistent/empty lengths")
            if not np.all(np.isfinite(np.asarray(paired["episode_return"]))):
              errors.append("paired returns contain non-finite values")
            if (
                args.require_reference_probe and
                not np.any(np.asarray(paired["is_reference"], dtype=bool))
            ):
              errors.append("paired returns contain no K=128 reference rows")
      except Exception as exc:
        errors.append(f"cannot validate paired returns: {exc}")
  report["csv_checks"] = {
      key: {"valid": value[0], "rows": value[1], "error": value[2]}
      for key, value in checks.items()
  }
  errors.extend(value[2] for value in checks.values() if not value[0])
  if args.controller == "adaptive":
    coverage_errors = validate_adaptive_coverage(run_dir, args)
    errors.extend(coverage_errors)
    report["coverage_errors"] = coverage_errors

  gif_index_path = run_dir / "artifacts/gifs/index.json"
  if not nonempty(gif_index_path):
    errors.append(f"missing GIF index {gif_index_path}")
    gif_index: dict[str, object] = {}
  else:
    try:
      gif_index = json.loads(gif_index_path.read_text())
    except Exception as exc:
      errors.append(f"invalid GIF index: {exc}")
      gif_index = {}
  indexed_steps = set()
  for item in gif_index.get("gifs", []) if isinstance(gif_index, dict) else []:
    try:
      step = int(item["step"])
      gif_path = run_dir / str(item["path"])
    except Exception as exc:
      errors.append(f"invalid GIF entry {item!r}: {exc}")
      continue
    indexed_steps.add(step)
    if not nonempty(gif_path):
      errors.append(f"missing or empty GIF {gif_path}")
    metadata_path = run_dir / str(item.get("metadata_path", ""))
    if not nonempty(metadata_path):
      errors.append(f"missing rollout metadata {metadata_path}")
      continue
    try:
      metadata = json.loads(metadata_path.read_text())
      trajectories = metadata.get("trajectories", {})
      if set(trajectories) != {"delay0", "delay4"}:
        errors.append(
            f"anchor {step} conditions are {sorted(trajectories)}, "
            "expected delay0/delay4"
        )
      for condition, expected_delay in (("delay0", 0), ("delay4", 4)):
        record = trajectories.get(condition, {})
        if record.get("effective_action_delay_at_reset") != expected_delay:
          errors.append(
              f"anchor {step} {condition} delay metadata is invalid"
          )
        trajectory_path = metadata_path.parent / str(record.get("path", ""))
        if not nonempty(trajectory_path):
          errors.append(f"missing trajectory {trajectory_path}")
    except Exception as exc:
      errors.append(f"invalid rollout metadata at anchor {step}: {exc}")
  missing_anchors = sorted(set(anchors) - indexed_steps)
  if missing_anchors:
    errors.append(f"GIF index missing anchors {missing_anchors}")
  missing_checkpoints = [
      step for step in anchors
      if not (run_dir / "artifacts/anchor_checkpoints" / str(step)).is_dir()
  ]
  if missing_checkpoints:
    errors.append(f"anchor checkpoints missing steps {missing_checkpoints}")
  report["gif_anchor_steps"] = sorted(indexed_steps)
  report["checkpoint_anchor_steps"] = [
      step for step in anchors if step not in missing_checkpoints
  ]
  report["valid"] = not errors
  report["errors"] = errors
  print(json.dumps(report, indent=2, sort_keys=True))
  if errors:
    print("INVALID")
    raise SystemExit(2)
  print("VALID")


if __name__ == "__main__":
  main()
