#!/usr/bin/env python3
"""Build terminal analysis artifacts for one Cartpole delay-pilot run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np


ROUGHNESS_RE = re.compile(
    r"^(?P<reference>reference_probe/)?dense_rhs/candidate_"
    r"(?P<horizon>\d+)_roughness_m(?P<count>2|4|8|16|32|64)$"
)
ALL_ROUGHNESS_RE = re.compile(
    r"^(?P<reference>reference_probe/)?dense_rhs/horizon_"
    r"(?P<horizon>\d+)_roughness_m(?P<count>2|4|8|16|32|64)$"
)
PROJECTION_RE = re.compile(
    r"^(?P<reference>reference_probe/)?dense_rhs/horizon_"
    r"(?P<horizon>\d+)_roughness_projection_(?P<direction>\d+)$"
)
RETURN_RE = re.compile(
    r"^(?:(?P<source>reference_probe|conditional_reference_probe)/)?"
    r"dense_rhs/candidate_"
    r"(?P<horizon>\d+)_return_replica_(?P<replica>\d+)$"
)
CANDIDATE_STAT_RE = re.compile(
    r"^(?P<reference>reference_probe/)?dense_rhs/candidate_"
    r"(?P<horizon>\d+)_(?P<stat>env_mean|env_std)$"
)
CONDITIONAL_REFERENCE_STAT_RE = re.compile(
    r"^conditional_reference_probe/dense_rhs/candidate_(?P<horizon>\d+)_"
    r"(?P<stat>env_mean|env_std)$"
)
RETURN_COUNTS = (8, 16, 32, 64, 128)


def return_source(match: re.Match) -> str:
  prefix = match.group("source")
  return {
      None: "deployed",
      "reference_probe": "reference",
      "conditional_reference_probe": "conditional_reference",
  }[prefix]


def utc_now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(temporary, path)


def read_scalars(path: Path) -> list[dict[str, str]]:
  if not path.is_file() or path.stat().st_size == 0:
    raise FileNotFoundError(f"missing scalar log: {path}")
  with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
  if not rows:
    raise ValueError(f"scalar log has no data rows: {path}")
  for row in rows:
    step = int(row["step"])
    value = float(row["value"])
    if step < 0 or not math.isfinite(value):
      raise ValueError(f"invalid scalar row: {row}")
  return rows


def scalar_lookup(rows: list[dict[str, str]]) -> dict[tuple[int, str], float]:
  return {
      (int(row["step"]), row["tag"]): float(row["value"])
      for row in rows
  }


def _influence_se(influence: np.ndarray) -> np.ndarray:
  sample_count = influence.shape[-1]
  if sample_count < 2:
    return np.zeros(influence.shape[:-1], dtype=np.float64)
  centred = influence - np.mean(influence, axis=-1, keepdims=True)
  return np.sqrt(np.maximum(
      np.sum(np.square(centred), axis=-1) /
      (sample_count * (sample_count - 1)),
      0.0,
  ))


def reconstruct_decision(*,
                         horizons: list[int],
                         incumbent: int,
                         paired_returns: dict[int, list[float]],
                         projections: dict[int, list[float]],
                         score_mode: str,
                         probe_count: int,
                         return_count: int | None = None,
                         confidence_z: float = 1.6448536,
                         switch_threshold: float = 0.02,
                         evidence_floor: float = 1e-6,
                         return_scale: float = 50.0,
                         return_std_scale: float = 50.0,
                         log_roughness_scale: float = 1.0) -> dict | None:
  """Reconstructs the exact paired-LCB selector for nested M/K evidence."""
  if incumbent not in horizons or probe_count < 2:
    return None
  available_k = min(len(paired_returns.get(horizon, ())) for horizon in horizons)
  k = available_k if return_count is None else min(int(return_count), available_k)
  if k < 2 or any(len(projections.get(horizon, ())) < probe_count for horizon in horizons):
    return None
  returns = np.asarray(
      [paired_returns[horizon][:k] for horizon in horizons], dtype=np.float64
  )
  direction_values = np.asarray(
      [projections[horizon][:probe_count] for horizon in horizons],
      dtype=np.float64,
  )
  mean = np.mean(returns, axis=-1)
  std = np.std(returns, axis=-1, ddof=0)
  roughness = np.sqrt(np.mean(np.square(direction_values), axis=-1) + 1e-6)
  incumbent_idx = horizons.index(incumbent)
  floor = float(evidence_floor)
  if score_mode == "additive":
    score = (
        (mean - mean[incumbent_idx]) / return_scale -
        (std - std[incumbent_idx]) / return_std_scale -
        (
            np.log(np.maximum(roughness, floor)) -
            np.log(max(roughness[incumbent_idx], floor))
        ) / log_roughness_scale
    )
  elif score_mode == "multiplicative":
    score = (
        np.log(np.maximum(mean, floor)) -
        np.log(max(mean[incumbent_idx], floor)) -
        (
            np.log(np.maximum(std, floor)) -
            np.log(max(std[incumbent_idx], floor))
        ) -
        (
            np.log(np.maximum(roughness, floor)) -
            np.log(max(roughness[incumbent_idx], floor))
        )
    )
  else:
    return None

  centred_returns = returns - mean[:, None]
  influence_mean = centred_returns
  influence_std = (
      np.square(centred_returns) - np.square(std[:, None])
  ) / (2.0 * np.maximum(std[:, None], floor))
  if score_mode == "additive":
    return_influence = (
        (influence_mean - influence_mean[incumbent_idx]) / return_scale -
        (influence_std - influence_std[incumbent_idx]) / return_std_scale
    )
  else:
    return_influence = (
        influence_mean / np.maximum(mean, floor)[:, None] -
        influence_mean[incumbent_idx] / max(mean[incumbent_idx], floor) -
        influence_std / np.maximum(std, floor)[:, None] +
        influence_std[incumbent_idx] / max(std[incumbent_idx], floor)
    )
  return_se = _influence_se(return_influence)

  squared = np.square(direction_values)
  mean_squared = np.mean(squared, axis=-1, keepdims=True)
  influence_rms = (squared - mean_squared) / (
      2.0 * np.maximum(roughness, floor)[:, None]
  )
  influence_log_roughness = influence_rms / np.maximum(
      roughness, floor
  )[:, None]
  roughness_influence = -(
      influence_log_roughness - influence_log_roughness[incumbent_idx]
  )
  if score_mode == "additive":
    roughness_influence /= log_roughness_scale
  roughness_se = _influence_se(roughness_influence)
  score_se = np.sqrt(np.square(return_se) + np.square(roughness_se))
  score_se[incumbent_idx] = 0.0
  lcb = score - confidence_z * score_se
  proposed_idx = int(np.argmax(lcb))
  proposed = horizons[proposed_idx]
  switch = proposed != incumbent and lcb[proposed_idx] > switch_threshold
  selected = incumbent
  if switch:
    selected = incumbent + int(np.clip(proposed - incumbent, -1, 1))
  return {
      "mean": mean,
      "std": std,
      "roughness": roughness,
      "score": score,
      "score_se": score_se,
      "score_lcb": lcb,
      "proposed_horizon": proposed,
      "selected_horizon": selected,
      "switch": bool(switch),
      "return_count": k,
      "probe_count": probe_count,
      "mean_floor_hits": int(np.sum(mean <= floor)),
      "std_floor_hits": int(np.sum(std <= floor)),
      "roughness_floor_hits": int(np.sum(roughness <= floor)),
  }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  os.replace(temporary, path)


def build_probe_calibration(metrics_dir: Path,
                            scalar_rows: list[dict[str, str]]) -> int:
  estimates: dict[tuple[int, int, int, bool], float] = {}
  for row in scalar_rows:
    match = ROUGHNESS_RE.fullmatch(row["tag"])
    if match is None:
      continue
    estimates[
        (
            int(row["step"]),
            int(match.group("horizon")),
            int(match.group("count")),
            bool(match.group("reference")),
        )
    ] = float(row["value"])
  lookup = scalar_lookup(scalar_rows)
  output = []
  groups = sorted({(step, reference) for step, _, _, reference in estimates})
  for step, reference in groups:
    horizons = sorted({
        horizon
        for candidate_step, horizon, _, candidate_reference in estimates
        if candidate_step == step and candidate_reference == reference
    })
    counts = sorted({
        count
        for candidate_step, _, count, candidate_reference in estimates
        if candidate_step == step and candidate_reference == reference
    })
    full_values = {
        horizon: estimates.get((step, horizon, 64, reference), math.nan)
        for horizon in horizons
    }
    full_rank = {
        horizon: rank + 1
        for rank, horizon in enumerate(
            sorted(horizons, key=lambda item: (full_values[item], item))
        )
        if math.isfinite(full_values[horizon])
    }
    prefix = "reference_probe/" if reference else ""
    selected = lookup.get((step, prefix + "dense_rhs/selected_horizon"), math.nan)
    proposed = lookup.get((step, prefix + "dense_rhs/proposed_horizon"), math.nan)
    previous_tag = (
        "reference_probe/incumbent_horizon"
        if reference else "dense_rhs/previous_horizon"
    )
    previous = lookup.get((step, previous_tag), math.nan)
    for count in counts:
      count_values = {
          horizon: estimates.get((step, horizon, count, reference), math.nan)
          for horizon in horizons
      }
      count_rank = {
          horizon: rank + 1
          for rank, horizon in enumerate(
              sorted(horizons, key=lambda item: (count_values[item], item))
          )
          if math.isfinite(count_values[horizon])
      }
      for horizon in horizons:
        estimate = count_values[horizon]
        reference_value = full_values[horizon]
        if not (math.isfinite(estimate) and math.isfinite(reference_value)):
          continue
        output.append({
            "step": step,
            "source": "reference_query" if reference else "deployed_query",
            "candidate_horizon": horizon,
            "probe_count": count,
            "roughness": estimate,
            "roughness_m64": reference_value,
            "absolute_error_vs_m64": abs(estimate - reference_value),
            "relative_error_vs_m64": (
                abs(estimate - reference_value) / max(abs(reference_value), 1e-12)
            ),
            "roughness_rank": count_rank[horizon],
            "roughness_rank_m64": full_rank[horizon],
            "rank_matches_m64": int(count_rank[horizon] == full_rank[horizon]),
            "selected_horizon": selected,
            "proposed_horizon": proposed,
            "previous_horizon": previous,
            "switch": (
                int(selected != previous)
                if math.isfinite(selected) and math.isfinite(previous) else ""
            ),
        })
  write_csv(
      metrics_dir / "probe_calibration.csv",
      [
          "step", "source", "candidate_horizon", "probe_count", "roughness",
          "roughness_m64", "absolute_error_vs_m64", "relative_error_vs_m64",
          "roughness_rank", "roughness_rank_m64", "rank_matches_m64",
          "selected_horizon", "proposed_horizon", "previous_horizon", "switch",
      ],
      output,
  )
  return len(output)


def build_all_horizon_roughness_artifacts(
    metrics_dir: Path,
    scalar_rows: list[dict[str, str]],
) -> tuple[int, int]:
  nested = {}
  projections = {}
  for row in scalar_rows:
    nested_match = ALL_ROUGHNESS_RE.fullmatch(row["tag"])
    if nested_match is not None:
      nested[
          (
              int(row["step"]),
              int(nested_match.group("horizon")),
              int(nested_match.group("count")),
              bool(nested_match.group("reference")),
          )
      ] = float(row["value"])
      continue
    projection_match = PROJECTION_RE.fullmatch(row["tag"])
    if projection_match is not None:
      key = (
          int(row["step"]),
          int(projection_match.group("horizon")),
          int(projection_match.group("direction")),
          bool(projection_match.group("reference")),
      )
      projections[key] = float(row["value"])
  output = []
  for (step, horizon, count, reference), estimate in sorted(nested.items()):
    full = nested.get((step, horizon, 64, reference), math.nan)
    if not math.isfinite(full):
      continue
    log_ratio = math.log(max(abs(estimate), 1e-12)) - math.log(
        max(abs(full), 1e-12)
    )
    output.append({
        "step": step,
        "source": "reference_query" if reference else "deployed_query",
        "effective_training_delay": 4 if 150_000 <= step < 350_000 else 0,
        "horizon": horizon,
        "probe_count": count,
        "roughness": estimate,
        "roughness_m64": full,
        "absolute_error_vs_m64": abs(estimate - full),
        "symmetric_relative_error_vs_m64": (
            2.0 * abs(estimate - full) /
            max(abs(estimate) + abs(full), 1e-12)
        ),
        "log_ratio_vs_m64": log_ratio,
    })
  write_csv(
      metrics_dir / "roughness_all_horizons.csv",
      [
          "step", "source", "effective_training_delay", "horizon",
          "probe_count", "roughness", "roughness_m64",
          "absolute_error_vs_m64", "symmetric_relative_error_vs_m64",
          "log_ratio_vs_m64",
      ],
      output,
  )
  projection_path = metrics_dir / "roughness_projections.npz"
  projection_records = [
      (step, horizon, direction, value, reference)
      for (step, horizon, direction, reference), value
      in sorted(projections.items())
  ]
  temporary = projection_path.with_suffix(".npz.tmp")
  with temporary.open("wb") as handle:
    np.savez_compressed(
        handle,
        step=np.asarray([item[0] for item in projection_records], dtype=np.int64),
        horizon=np.asarray([item[1] for item in projection_records], dtype=np.int32),
        direction=np.asarray([item[2] for item in projection_records], dtype=np.int32),
        projection=np.asarray([item[3] for item in projection_records], dtype=np.float32),
        is_reference=np.asarray([item[4] for item in projection_records], dtype=bool),
    )
  os.replace(temporary, projection_path)
  return len(output), len(projection_records)


def build_roughness_bootstrap(metrics_dir: Path,
                              scalar_rows: list[dict[str, str]],
                              bootstrap_replicates: int = 1000) -> int:
  """Bootstraps directional-probe RMS uncertainty without treating M64 as truth."""
  grouped: dict[tuple[int, int, str], dict[int, float]] = {}
  for row in scalar_rows:
    match = PROJECTION_RE.fullmatch(row["tag"])
    if match is None:
      continue
    source = "reference_query" if match.group("reference") else "deployed_query"
    key = (int(row["step"]), int(match.group("horizon")), source)
    grouped.setdefault(key, {})[int(match.group("direction"))] = float(
        row["value"]
    )
  output = []
  for (step, horizon, source), values_by_direction in sorted(grouped.items()):
    ordered = np.asarray(
        [value for _, value in sorted(values_by_direction.items())],
        dtype=np.float64,
    )
    for count in (2, 4, 8, 16, 32, 64):
      if ordered.size < count:
        continue
      prefix = ordered[:count]
      seed_bytes = hashlib.sha256(
          f"{source}:{step}:{horizon}:{count}".encode()
      ).digest()[:8]
      rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
      indices = rng.integers(
          0,
          count,
          size=(int(bootstrap_replicates), count),
      )
      bootstrap = np.sqrt(
          np.mean(np.square(prefix[indices]), axis=1) + 1e-6
      )
      estimate = float(np.sqrt(np.mean(np.square(prefix)) + 1e-6))
      output.append({
          "step": step,
          "source": source,
          "candidate_horizon": horizon,
          "probe_count": count,
          "roughness": estimate,
          "bootstrap_replicates": int(bootstrap_replicates),
          "bootstrap_se": float(np.std(bootstrap, ddof=1)),
          "bootstrap_lcb95": float(np.quantile(bootstrap, 0.025)),
          "bootstrap_ucb95": float(np.quantile(bootstrap, 0.975)),
          "bootstrap_interval_width": float(
              np.quantile(bootstrap, 0.975) - np.quantile(bootstrap, 0.025)
          ),
      })
  write_csv(
      metrics_dir / "roughness_bootstrap.csv",
      [
          "step", "source", "candidate_horizon", "probe_count",
          "roughness", "bootstrap_replicates", "bootstrap_se",
          "bootstrap_lcb95", "bootstrap_ucb95", "bootstrap_interval_width",
      ],
      output,
  )
  return len(output)


def _paired_and_projection_maps(scalar_rows: list[dict[str, str]]):
  returns: dict[tuple[int, int, str], dict[int, float]] = {}
  projections: dict[tuple[int, int, str], dict[int, float]] = {}
  for row in scalar_rows:
    return_match = RETURN_RE.fullmatch(row["tag"])
    if return_match is not None:
      key = (
          int(row["step"]),
          int(return_match.group("horizon")),
          return_source(return_match),
      )
      returns.setdefault(key, {})[int(return_match.group("replica"))] = float(
          row["value"]
      )
      continue
    projection_match = PROJECTION_RE.fullmatch(row["tag"])
    if projection_match is not None:
      key = (
          int(row["step"]),
          int(projection_match.group("horizon")),
          "reference" if projection_match.group("reference") else "deployed",
      )
      projections.setdefault(key, {})[
          int(projection_match.group("direction"))
      ] = float(row["value"])
  return returns, projections


def build_nested_decision_calibration(metrics_dir: Path,
                                      scalar_rows: list[dict[str, str]],
                                      score_mode: str) -> tuple[int, int]:
  if score_mode not in {"additive", "multiplicative"}:
    return 0, 0
  lookup = scalar_lookup(scalar_rows)
  returns_map, projection_map = _paired_and_projection_maps(scalar_rows)
  candidate_groups = {}
  for row in scalar_rows:
    match = ROUGHNESS_RE.fullmatch(row["tag"])
    if match is None:
      continue
    candidate_groups.setdefault(
        (int(row["step"]), bool(match.group("reference"))), set()
    ).add(int(match.group("horizon")))

  m_rows = []
  k_rows = []
  for (step, reference), horizon_set in sorted(candidate_groups.items()):
    horizons = sorted(horizon_set)
    incumbent_tag = (
        "reference_probe/incumbent_horizon"
        if reference else "dense_rhs/previous_horizon"
    )
    incumbent_value = lookup.get((step, incumbent_tag))
    if incumbent_value is None:
      continue
    incumbent = int(incumbent_value)
    paired = {}
    projections = {}
    complete = True
    source = "reference" if reference else "deployed"
    for horizon in horizons:
      return_values = returns_map.get((step, horizon, source), {})
      projection_values = projection_map.get((step, horizon, source), {})
      if not return_values or not projection_values:
        complete = False
        break
      paired[horizon] = [
          value for _, value in sorted(return_values.items())
      ]
      projections[horizon] = [
          value for _, value in sorted(projection_values.items())
      ]
    if not complete:
      continue
    full_m = reconstruct_decision(
        horizons=horizons,
        incumbent=incumbent,
        paired_returns=paired,
        projections=projections,
        score_mode=score_mode,
        probe_count=64,
    )
    if full_m is None:
      continue
    delay = 4 if 150_000 <= step < 350_000 else 0
    for probe_count in (2, 4, 8, 16, 32, 64):
      nested = reconstruct_decision(
          horizons=horizons,
          incumbent=incumbent,
          paired_returns=paired,
          projections=projections,
          score_mode=score_mode,
          probe_count=probe_count,
      )
      if nested is None:
        continue
      m_rows.append({
          "step": step,
          "source": "reference_query" if reference else "deployed_query",
          "effective_training_delay": delay,
          "score_mode": score_mode,
          "probe_count": probe_count,
          "return_replicas": nested["return_count"],
          "incumbent_horizon": incumbent,
          "proposed_horizon": nested["proposed_horizon"],
          "selected_horizon": nested["selected_horizon"],
          "switch": int(nested["switch"]),
          "proposed_horizon_m64": full_m["proposed_horizon"],
          "selected_horizon_m64": full_m["selected_horizon"],
          "switch_m64": int(full_m["switch"]),
          "proposal_matches_m64": int(
              nested["proposed_horizon"] == full_m["proposed_horizon"]
          ),
          "selection_matches_m64": int(
              nested["selected_horizon"] == full_m["selected_horizon"]
          ),
          "switch_matches_m64": int(nested["switch"] == full_m["switch"]),
          "mean_floor_hits": nested["mean_floor_hits"],
          "std_floor_hits": nested["std_floor_hits"],
          "roughness_floor_hits": nested["roughness_floor_hits"],
          "min_return_mean": float(np.min(nested["mean"])),
          "median_return_mean": float(np.median(nested["mean"])),
          "min_return_std": float(np.min(nested["std"])),
          "median_return_std": float(np.median(nested["std"])),
          "min_roughness": float(np.min(nested["roughness"])),
          "median_roughness": float(np.median(nested["roughness"])),
      })

    if reference:
      full_k = full_m
      for return_count in RETURN_COUNTS:
        nested_k = reconstruct_decision(
            horizons=horizons,
            incumbent=incumbent,
            paired_returns=paired,
            projections=projections,
            score_mode=score_mode,
            probe_count=64,
            return_count=return_count,
        )
        if nested_k is None:
          continue
        k_rows.append({
            "step": step,
            "effective_training_delay": delay,
            "score_mode": score_mode,
            "return_replicas": nested_k["return_count"],
            "probe_count": 64,
            "incumbent_horizon": incumbent,
            "proposed_horizon": nested_k["proposed_horizon"],
            "selected_horizon": nested_k["selected_horizon"],
            "switch": int(nested_k["switch"]),
            "proposed_horizon_k128": full_k["proposed_horizon"],
            "selected_horizon_k128": full_k["selected_horizon"],
            "switch_k128": int(full_k["switch"]),
            "proposal_matches_k128": int(
                nested_k["proposed_horizon"] == full_k["proposed_horizon"]
            ),
            "selection_matches_k128": int(
                nested_k["selected_horizon"] == full_k["selected_horizon"]
            ),
            "switch_matches_k128": int(nested_k["switch"] == full_k["switch"]),
            "mean_floor_hits": nested_k["mean_floor_hits"],
            "std_floor_hits": nested_k["std_floor_hits"],
            "roughness_floor_hits": nested_k["roughness_floor_hits"],
        })
  m_fields = [
      "step", "source", "effective_training_delay", "score_mode",
      "probe_count", "return_replicas", "incumbent_horizon",
      "proposed_horizon", "selected_horizon", "switch",
      "proposed_horizon_m64", "selected_horizon_m64", "switch_m64",
      "proposal_matches_m64", "selection_matches_m64", "switch_matches_m64",
      "mean_floor_hits", "std_floor_hits", "roughness_floor_hits",
      "min_return_mean", "median_return_mean", "min_return_std",
      "median_return_std", "min_roughness", "median_roughness",
  ]
  k_fields = [
      "step", "effective_training_delay", "score_mode", "return_replicas",
      "probe_count", "incumbent_horizon", "proposed_horizon",
      "selected_horizon", "switch", "proposed_horizon_k128",
      "selected_horizon_k128", "switch_k128", "proposal_matches_k128",
      "selection_matches_k128", "switch_matches_k128", "mean_floor_hits",
      "std_floor_hits", "roughness_floor_hits",
  ]
  write_csv(metrics_dir / "decision_probe_calibration.csv", m_fields, m_rows)
  write_csv(metrics_dir / "return_decision_calibration.csv", k_fields, k_rows)
  return len(m_rows), len(k_rows)


def build_conditional_return_reference(metrics_dir: Path,
                                       scalar_rows: list[dict[str, str]],
                                       rollout_steps: int) -> int:
  """Summarizes the K=32 full-horizon deployed-planner shadow evaluation.

  This is conditional on the learned model checkpoint and shared rollout
  draws. It is not a population oracle. Paired gaps and their normal CIs make
  the finite-K uncertainty explicit.
  """
  values = {}
  replica_values: dict[tuple[int, int], dict[int, float]] = {}
  for row in scalar_rows:
    match = CONDITIONAL_REFERENCE_STAT_RE.fullmatch(row["tag"])
    if match is not None:
      values[
          (int(row["step"]), int(match.group("horizon")), match.group("stat"))
      ] = float(row["value"])
      continue
    return_match = RETURN_RE.fullmatch(row["tag"])
    if (
        return_match is not None and
        return_source(return_match) == "conditional_reference"
    ):
      key = (int(row["step"]), int(return_match.group("horizon")))
      replica_values.setdefault(key, {})[
          int(return_match.group("replica"))
      ] = float(row["value"])
  output = []
  for step in sorted({key[0] for key in values}):
    horizons = sorted({key[1] for key in values if key[0] == step})
    means = {
        horizon: values.get((step, horizon, "env_mean"), math.nan)
        for horizon in horizons
    }
    finite_horizons = [
        horizon for horizon in horizons if math.isfinite(means[horizon])
    ]
    if not finite_horizons:
      continue
    best_horizon = max(finite_horizons, key=lambda h: (means[h], -h))
    ordered_returns = {}
    for horizon in finite_horizons:
      records = replica_values.get((step, horizon), {})
      ordered_returns[horizon] = np.asarray(
          [value for _, value in sorted(records.items())],
          dtype=np.float64,
      )
    replica_counts = {array.size for array in ordered_returns.values()}
    paired = len(replica_counts) == 1 and next(iter(replica_counts), 0) >= 2
    if paired:
      return_matrix = np.stack(
          [ordered_returns[horizon] for horizon in finite_horizons], axis=0
      )
      winners = np.argmax(return_matrix, axis=0)
      best_index = finite_horizons.index(best_horizon)
      best_returns = return_matrix[best_index]
    for horizon_index, horizon in enumerate(finite_horizons):
      if paired:
        gaps = best_returns - return_matrix[horizon_index]
        gap_mean = float(np.mean(gaps))
        gap_se = float(np.std(gaps, ddof=1) / np.sqrt(gaps.size))
        gap_lcb = gap_mean - 1.959964 * gap_se
        gap_ucb = gap_mean + 1.959964 * gap_se
        probability_replica_best = float(np.mean(winners == horizon_index))
        replicas = int(gaps.size)
      else:
        gap_mean = gap_se = gap_lcb = gap_ucb = math.nan
        probability_replica_best = math.nan
        replicas = 0
      output.append({
          "step": step,
          "effective_training_delay": 4 if 150_000 <= step < 350_000 else 0,
          "rollout_steps": int(rollout_steps),
          "planner": "deployed_512_24_64_6",
          "return_replicas": replicas,
          "candidate_horizon": horizon,
          "return_mean": means[horizon],
          "return_population_std": values.get(
              (step, horizon, "env_std"), math.nan
          ),
          "best_by_sample_mean_horizon": best_horizon,
          "best_by_sample_mean_return": means[best_horizon],
          "paired_gap_to_best_mean": gap_mean,
          "paired_gap_to_best_se": gap_se,
          "paired_gap_to_best_lcb95": gap_lcb,
          "paired_gap_to_best_ucb95": gap_ucb,
          "probability_replica_best": probability_replica_best,
      })
  write_csv(
      metrics_dir / "conditional_return_reference.csv",
      [
          "step", "effective_training_delay", "rollout_steps", "planner",
          "return_replicas", "candidate_horizon", "return_mean",
          "return_population_std", "best_by_sample_mean_horizon",
          "best_by_sample_mean_return", "paired_gap_to_best_mean",
          "paired_gap_to_best_se", "paired_gap_to_best_lcb95",
          "paired_gap_to_best_ucb95", "probability_replica_best",
      ],
      output,
  )
  return len(output)


def build_return_artifacts(metrics_dir: Path,
                           scalar_rows: list[dict[str, str]]) -> tuple[int, int]:
  grouped: dict[tuple[int, int, str], dict[int, float]] = {}
  for row in scalar_rows:
    match = RETURN_RE.fullmatch(row["tag"])
    if match is None:
      continue
    step = int(row["step"])
    horizon = int(match.group("horizon"))
    replica = int(match.group("replica"))
    source = return_source(match)
    value = float(row["value"])
    grouped.setdefault((step, horizon, source), {})[replica] = value
  records = [
      (step, horizon, replica, value, source)
      for (step, horizon, source), replica_values in sorted(grouped.items())
      for replica, value in sorted(replica_values.items())
  ]

  metrics_dir.mkdir(parents=True, exist_ok=True)
  npz_path = metrics_dir / "paired_returns.npz"
  temporary = npz_path.with_suffix(".npz.tmp")
  with temporary.open("wb") as handle:
    np.savez_compressed(
        handle,
        step=np.asarray([item[0] for item in records], dtype=np.int64),
        candidate_horizon=np.asarray([item[1] for item in records], dtype=np.int32),
        replica=np.asarray([item[2] for item in records], dtype=np.int32),
        episode_return=np.asarray([item[3] for item in records], dtype=np.float32),
        source=np.asarray([item[4] for item in records], dtype="U24"),
        is_reference=np.asarray(
            [item[4] == "reference" for item in records], dtype=bool
        ),
        is_conditional_reference=np.asarray(
            [item[4] == "conditional_reference" for item in records],
            dtype=bool,
        ),
    )
  os.replace(temporary, npz_path)

  calibration = []
  reference_groups = {
      key: values for key, values in grouped.items() if key[2] == "reference"
  }
  by_step: dict[int, list[int]] = {}
  for step, horizon, _ in reference_groups:
    by_step.setdefault(step, []).append(horizon)
  for step, horizons in sorted(by_step.items()):
    horizons = sorted(set(horizons))
    full_means = {}
    full_stds = {}
    for horizon in horizons:
      ordered = [
          value for _, value in sorted(
              reference_groups[(step, horizon, "reference")].items()
          )
      ]
      if len(ordered) >= 128:
        full_means[horizon] = float(np.mean(ordered[:128]))
        full_stds[horizon] = float(np.std(ordered[:128], ddof=0))
    full_ranks = {
        horizon: rank + 1
        for rank, horizon in enumerate(
            sorted(full_means, key=lambda item: (-full_means[item], item))
        )
    }
    for horizon in horizons:
      ordered = [
          value for _, value in sorted(
              reference_groups[(step, horizon, "reference")].items()
          )
      ]
      if len(ordered) < 128:
        continue
      for count in RETURN_COUNTS:
        values = np.asarray(ordered[:count], dtype=np.float64)
        nested_mean = float(np.mean(values))
        nested_std = float(np.std(values, ddof=0))
        count_means = {
            other: float(np.mean([
                value for _, value in sorted(
                    reference_groups[(step, other, "reference")].items()
                )
            ][:count]))
            for other in horizons
            if len(reference_groups[(step, other, "reference")]) >= count
        }
        ranks = {
            other: rank + 1
            for rank, other in enumerate(
                sorted(count_means, key=lambda item: (-count_means[item], item))
            )
        }
        calibration.append({
            "step": step,
            "candidate_horizon": horizon,
            "return_replicas": count,
            "return_mean": nested_mean,
            "return_population_std": nested_std,
            "return_mean_k128": full_means[horizon],
            "return_population_std_k128": full_stds[horizon],
            "absolute_error_vs_k128": abs(nested_mean - full_means[horizon]),
            "std_absolute_error_vs_k128": abs(
                nested_std - full_stds[horizon]
            ),
            "return_rank": ranks[horizon],
            "return_rank_k128": full_ranks[horizon],
            "rank_matches_k128": int(ranks[horizon] == full_ranks[horizon]),
        })
  write_csv(
      metrics_dir / "return_probe_calibration.csv",
      [
          "step", "candidate_horizon", "return_replicas", "return_mean",
          "return_population_std", "return_mean_k128",
          "return_population_std_k128", "absolute_error_vs_k128",
          "std_absolute_error_vs_k128",
          "return_rank", "return_rank_k128", "rank_matches_k128",
      ],
      calibration,
  )
  return len(records), len(calibration)


def build_gif_index(run_dir: Path, run_id: str, anchor_steps: tuple[int, ...]) -> int:
  rollout_root = run_dir / "artifacts" / "rollouts" / run_id
  gifs = []
  for step in anchor_steps:
    rollout_dir = rollout_root / f"step_{step:06d}"
    gif_path = rollout_dir / "cartpole_delay0_vs_delay4.gif"
    metadata_path = rollout_dir / "metadata.json"
    if not gif_path.is_file() or gif_path.stat().st_size == 0:
      raise FileNotFoundError(f"missing rendered GIF: {gif_path}")
    if not metadata_path.is_file():
      raise FileNotFoundError(f"missing rollout metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    gifs.append({
        "step": step,
        "path": str(gif_path.relative_to(run_dir)),
        "metadata_path": str(metadata_path.relative_to(run_dir)),
        "selected_horizon": int(metadata["selected_horizon"]),
        "conditions": sorted(metadata.get("trajectories", {})),
    })
  atomic_json(
      run_dir / "artifacts" / "gifs" / "index.json",
      {"schema_version": 1, "run_id": run_id, "gifs": gifs},
  )
  return len(gifs)


def build_trajectory_summary(run_dir: Path,
                             run_id: str,
                             anchor_steps: tuple[int, ...]) -> int:
  rows = []
  rollout_root = run_dir / "artifacts" / "rollouts" / run_id
  for step in anchor_steps:
    rollout_dir = rollout_root / f"step_{step:06d}"
    for condition, expected_delay in (("delay0", 0), ("delay4", 4)):
      with np.load(
          rollout_dir / f"trajectory_{condition}.npz",
          allow_pickle=False,
      ) as trajectory:
        rewards = np.asarray(trajectory["reward"], dtype=np.float64)
        done = np.asarray(trajectory["done"], dtype=bool)
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        commanded = np.asarray(
            trajectory["commanded_action"], dtype=np.float64
        )
        applied = np.asarray(trajectory["applied_action"], dtype=np.float64)
        queue = np.asarray(trajectory["delayed_actions"], dtype=np.float64)
        delays = np.asarray(trajectory["effective_action_delay"], dtype=np.int32)
      if rewards.ndim == 1:
        rewards = rewards[:, None]
        done = done[:, None]
      for initial_state in range(rewards.shape[1]):
        completed = np.flatnonzero(done[:, initial_state])
        length = int(completed[0]) + 1 if completed.size else rewards.shape[0]
        positions = qpos[:length + 1, initial_state]
        pole_angle = positions[:, 1]
        angle_error = np.abs(np.arctan2(np.sin(pole_angle), np.cos(pole_angle)))
        rows.append({
            "step": step,
            "condition": condition,
            "effective_delay": expected_delay,
            "initial_state": initial_state,
            "episode_length": length,
            "episode_return": float(np.sum(rewards[:length, initial_state])),
            "upright_fraction": float(np.mean(np.cos(pole_angle) >= 0.8)),
            "mean_abs_pole_angle_rad": float(np.mean(angle_error)),
            "max_abs_cart_position": float(np.max(np.abs(positions[:, 0]))),
            "mean_abs_command_applied_mismatch": float(np.mean(np.abs(
                commanded[:length, initial_state] -
                applied[:length, initial_state]
            ))),
            "mean_queue_l2": float(np.mean(np.linalg.norm(
                queue[:length, initial_state].reshape(length, -1), axis=-1
            ))),
            "observed_delay_matches_condition": int(
                np.all(delays[:length + 1, initial_state] == expected_delay)
            ),
        })
  write_csv(
      run_dir / "metrics" / "trajectory_summary.csv",
      [
          "step", "condition", "effective_delay", "initial_state",
          "episode_length", "episode_return", "upright_fraction",
          "mean_abs_pole_angle_rad", "max_abs_cart_position",
          "mean_abs_command_applied_mismatch", "mean_queue_l2",
          "observed_delay_matches_condition",
      ],
      rows,
  )
  return len(rows)


def git_commit(repo: Path) -> str:
  result = subprocess.run(
      ["git", "-C", str(repo), "rev-parse", "HEAD"],
      check=True,
      text=True,
      capture_output=True,
  )
  return result.stdout.strip()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--controller", choices=("adaptive", "scripted", "fixed"), required=True)
  parser.add_argument("--score-mode", required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--expected-step", type=int, default=500_000)
  parser.add_argument(
      "--anchor-steps",
      default="0,100000,150000,250000,350000,450000,500000",
  )
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--config-hash", required=True)
  parser.add_argument(
      "--conditional-reference-eval-steps", type=int, default=500
  )
  args = parser.parse_args()
  run_dir = args.run_dir.resolve()
  repo = Path(__file__).resolve().parents[1]
  actual_commit = git_commit(repo)
  if actual_commit != args.expected_commit:
    raise ValueError(
        f"checkout changed during run: expected {args.expected_commit}, "
        f"found {actual_commit}"
    )
  anchors = tuple(sorted({int(value) for value in args.anchor_steps.split(",")}))
  scalar_rows = read_scalars(run_dir / "metrics" / "scalars.csv")
  max_logged_step = max(int(row["step"]) for row in scalar_rows)
  if max_logged_step < args.expected_step:
    raise ValueError(
        f"maximum logged step {max_logged_step} is below {args.expected_step}"
    )
  lookup = scalar_lookup(scalar_rows)
  final_eval_mean = lookup.get((args.expected_step, "eval/return_mean"), math.nan)
  final_eval_std = lookup.get((args.expected_step, "eval/return_std"), math.nan)
  if not (math.isfinite(final_eval_mean) and math.isfinite(final_eval_std)):
    raise ValueError("missing finite final evaluation mean/std")

  probe_rows = 0
  all_horizon_roughness_rows = 0
  roughness_projection_rows = 0
  roughness_bootstrap_rows = 0
  decision_probe_rows = 0
  return_decision_rows = 0
  conditional_reference_rows = 0
  paired_rows = 0
  return_calibration_rows = 0
  if args.controller == "adaptive":
    probe_rows = build_probe_calibration(run_dir / "metrics", scalar_rows)
    (
        all_horizon_roughness_rows,
        roughness_projection_rows,
    ) = build_all_horizon_roughness_artifacts(
        run_dir / "metrics", scalar_rows
    )
    roughness_bootstrap_rows = build_roughness_bootstrap(
        run_dir / "metrics", scalar_rows
    )
    decision_probe_rows, return_decision_rows = (
        build_nested_decision_calibration(
            run_dir / "metrics", scalar_rows, args.score_mode
        )
    )
    conditional_reference_rows = build_conditional_return_reference(
        run_dir / "metrics",
        scalar_rows,
        args.conditional_reference_eval_steps,
    )
    paired_rows, return_calibration_rows = build_return_artifacts(
        run_dir / "metrics", scalar_rows
    )
    if probe_rows == 0 or paired_rows == 0:
      raise ValueError("adaptive run produced no probe or paired-return records")
  gif_count = build_gif_index(run_dir, args.run_id, anchors)
  trajectory_summary_rows = build_trajectory_summary(
      run_dir, args.run_id, anchors
  )
  config_path = run_dir / ".hydra" / "config.yaml"
  config_sha256 = (
      hashlib.sha256(config_path.read_bytes()).hexdigest()
      if config_path.is_file() else None
  )
  manifest = {
      "schema_version": 1,
      "status": "completed",
      "completed_at_utc": utc_now(),
      "run_id": args.run_id,
      "controller": args.controller,
      "score_mode": args.score_mode,
      "seed": args.seed,
      "final_step": args.expected_step,
      "max_logged_step": max_logged_step,
      "final_eval_mean": final_eval_mean,
      "final_eval_std": final_eval_std,
      "git_commit": actual_commit,
      "profile_config_hash": args.config_hash,
      "config_sha256": config_sha256,
      "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
      "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
      "artifacts": {
          "gif_count": gif_count,
          "probe_calibration_rows": probe_rows,
          "all_horizon_roughness_rows": all_horizon_roughness_rows,
          "roughness_projection_rows": roughness_projection_rows,
          "roughness_bootstrap_rows": roughness_bootstrap_rows,
          "decision_probe_calibration_rows": decision_probe_rows,
          "return_decision_calibration_rows": return_decision_rows,
          "conditional_return_reference_rows": conditional_reference_rows,
          "paired_return_rows": paired_rows,
          "return_calibration_rows": return_calibration_rows,
          "trajectory_summary_rows": trajectory_summary_rows,
      },
  }
  atomic_json(run_dir / "artifacts" / "run_manifest.json", manifest)
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
