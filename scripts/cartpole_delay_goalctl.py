#!/usr/bin/env python3
"""Single-writer steward for the frozen Cartpole delay pilot.

The controller is deliberately small and campaign-specific.  It validates an
explicit seven-profile queue, observes node-wide NCC GPU load, records Slurm
terminal states in an append-only ledger, and fills at most four physically
idle one-GPU slots.  Scientific failures are never retried at the same commit.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOAL = ROOT / "goals/cartpole_delay_pilot.yaml"
SSH_HELPER = Path(
    "/Users/ggmar/.codex/skills/remote-machines/scripts/ssh_host.sh"
)
REMOTE_PYTHON = '"$HOME/.venvs/temporalhorizon-jax/bin/python"'
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
TRANSIENT_STATES = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED"}
LEDGER_FIELDS = [
    "timestamp",
    "event_type",
    "run_id",
    "controller",
    "score_mode",
    "seed",
    "job_id",
    "attempt",
    "status",
    "slurm_state",
    "exit_code",
    "failure_class",
    "git_commit",
    "config_hash",
    "remote_run_dir",
    "final_eval_mean",
    "final_eval_std",
    "gifs_valid",
    "probe_artifacts_valid",
    "notes",
]
STEWARD_DIRTY_PATHS = {
    "experiments/cartpole_delay_pilot_ledger.csv",
    "experiments/cartpole_delay_pilot_decisions.md",
}
STEWARD_DIRTY_PREFIXES = ("runs/results/cartpole_delay_pilot/",)


def load_goal(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def utc_now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_local(*command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command,
      check=check,
      text=True,
      capture_output=True,
  )


def git(repo: Path, *args: str) -> str:
  return run_local("git", "-C", str(repo), *args).stdout.strip()


def scientific_dirty_paths() -> list[str]:
  dirty = []
  # Do not use git(), whose whole-output strip would remove the first
  # porcelain status column when the first entry has an unstaged-only status.
  status_output = run_local(
      "git", "-C", str(ROOT), "status", "--porcelain=v1"
  ).stdout
  for line in status_output.splitlines():
    path = line[3:].strip()
    if " -> " in path:
      path = path.split(" -> ", 1)[1]
    if path in STEWARD_DIRTY_PATHS:
      continue
    if any(path.startswith(prefix) for prefix in STEWARD_DIRTY_PREFIXES):
      continue
    dirty.append(path)
  return dirty


def remote(command: str, *, attempts: int = 3) -> str:
  last: Optional[subprocess.CompletedProcess] = None
  for attempt in range(attempts):
    last = run_local(
        str(SSH_HELPER),
        "ncc-gpu1",
        command,
        check=False,
    )
    if last.returncode == 0:
      # ssh_host.sh may print connection diagnostics on stderr, never stdout.
      return last.stdout.strip()
    if attempt + 1 < attempts:
      time.sleep(1.0 + attempt)
  assert last is not None
  raise RuntimeError(
      f"NCC command failed after {attempts} attempts: {last.stderr.strip()}"
  )


def ledger_path(goal: dict[str, Any]) -> Path:
  return ROOT / goal["tracking"]["ledger"]


def read_ledger(goal: dict[str, Any]) -> list[dict[str, str]]:
  path = ledger_path(goal)
  if not path.exists():
    return []
  with path.open(newline="") as handle:
    return list(csv.DictReader(handle))


def append_ledger(goal: dict[str, Any], values: dict[str, Any]) -> None:
  path = ledger_path(goal)
  path.parent.mkdir(parents=True, exist_ok=True)
  exists = path.exists() and path.stat().st_size > 0
  row = {field: values.get(field, "") for field in LEDGER_FIELDS}
  row["timestamp"] = row["timestamp"] or utc_now()
  with path.open("a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
    if not exists:
      writer.writeheader()
    writer.writerow(row)


def profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  return sorted(goal["profiles"], key=lambda item: int(item["priority"]))


def profile_hash(goal: dict[str, Any], profile: dict[str, Any]) -> str:
  payload = {"shared": goal["shared"], "profile": profile}
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()[:16]


def events_for(rows: Iterable[dict[str, str]], run_id: str) -> list[dict[str, str]]:
  return [row for row in rows if row.get("run_id") == run_id]


def latest_event(
    rows: Iterable[dict[str, str]], run_id: str
) -> Optional[dict[str, str]]:
  matching = events_for(rows, run_id)
  return matching[-1] if matching else None


def attempts_for(rows: Iterable[dict[str, str]], run_id: str) -> int:
  return sum(
      row.get("event_type") == "launched" for row in events_for(rows, run_id)
  )


def completed_ids(rows: Iterable[dict[str, str]]) -> set[str]:
  return {
      row["run_id"]
      for row in rows
      if row.get("event_type") == "completed" and row.get("status") == "valid"
  }


def open_jobs(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
  latest: dict[str, dict[str, str]] = {}
  for row in rows:
    run_id = row.get("run_id", "")
    if run_id:
      latest[run_id] = row
  return {
      run_id: row
      for run_id, row in latest.items()
      if row.get("event_type") == "launched" and row.get("job_id")
  }


def parse_gpu_rows(text: str) -> list[dict[str, int]]:
  rows: list[dict[str, int]] = []
  for line in text.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3 or not all(re.fullmatch(r"\d+", part) for part in parts):
      continue
    index, memory_used, utilization = map(int, parts)
    rows.append(
        {"index": index, "memory_used_mib": memory_used, "utilization_pct": utilization}
    )
  return rows


def gpu_snapshot(goal: dict[str, Any]) -> tuple[str, list[dict[str, int]], int]:
  raw = remote(
      "nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
      "--format=csv,noheader,nounits"
  )
  rows = parse_gpu_rows(raw)
  constraints = goal["constraints"]
  free = sum(
      row["memory_used_mib"] <= int(constraints["physical_gpu_free_mem_mib"])
      and row["utilization_pct"] <= int(constraints["physical_gpu_free_util_pct"])
      for row in rows
  )
  return raw, rows, int(free)


def normalize_slurm_state(value: str) -> str:
  """Normalizes Slurm suffixes and human-readable cancellation reasons."""
  return value.strip().upper().split()[0].split("+")[0]


def slurm_state(job_id: str) -> tuple[str, str, str]:
  output = remote(
      "sacct -X -n -P "
      f"-j {shlex.quote(job_id)} "
      "--format=JobIDRaw,State,ExitCode,Elapsed"
  )
  for line in output.splitlines():
    parts = line.strip().split("|")
    if len(parts) >= 4 and parts[0] == str(job_id):
      state = normalize_slurm_state(parts[1])
      return state, parts[2].strip(), parts[3].strip()
  queue = remote(
      f"squeue -h -j {shlex.quote(job_id)} -o '%T|%M'"
  ).strip()
  if queue:
    state, elapsed = (queue.split("|", 1) + [""])[:2]
    state = normalize_slurm_state(state)
    return state, "", elapsed.strip()
  return "UNKNOWN", "", ""


def remote_repo_state(goal: dict[str, Any]) -> dict[str, str]:
  path = shlex.quote(goal["repo"]["remote_path"])
  output = remote(
      f"test -d {path}/.git && cd {path} && "
      "printf 'commit=%s\\n' \"$(git rev-parse HEAD)\" && "
      "printf 'branch=%s\\n' \"$(git branch --show-current)\" && "
      "printf 'dirty=%s\\n' \"$(git status --porcelain | wc -l)\""
  )
  result: dict[str, str] = {}
  for line in output.splitlines():
    if "=" in line:
      key, value = line.split("=", 1)
      result[key.strip()] = value.strip()
  return result


def gate_state(goal: dict[str, Any]) -> dict[str, Any]:
  path = ROOT / goal["gates"]["status_file"]
  if not path.exists():
    return {"exists": False, "full_launch_ready": False}
  try:
    state = json.loads(path.read_text())
  except Exception as exc:
    return {"exists": True, "full_launch_ready": False, "error": str(exc)}
  state["exists"] = True
  return state


def validate(goal: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  items = profiles(goal)
  expected = int(goal["constraints"]["expected_profiles"])
  if len(items) != expected:
    errors.append(f"expected {expected} profiles, found {len(items)}")
  if [int(item.get("priority", -1)) for item in items] != list(
      range(1, expected + 1)
  ):
    errors.append("profile priorities must be exactly 1..7")
  if int(goal["constraints"].get("max_active_gpus", 0)) > 4:
    errors.append("max_active_gpus exceeds the four-GPU NCC allocation")
  ids = [item.get("run_id") for item in items]
  if len(ids) != len(set(ids)):
    errors.append("profile run_id values are not unique")
  matrix = {(item["controller"], item["score_mode"], int(item["seed"])) for item in items}
  required = {
      (method, method if method in {"additive", "multiplicative"} else "none", seed)
      for seed in (1, 23)
      for method in ("additive", "multiplicative", "scripted")
  }
  normalized = {
      (
          item["score_mode"] if item["controller"] == "adaptive" else item["controller"],
          item["score_mode"],
          int(item["seed"]),
      )
      for item in items
      if item["controller"] != "fixed"
  }
  if normalized != required:
    errors.append(f"six-run intervention matrix mismatch: {sorted(normalized)}")
  expected_delay = "0:0,150000:4,350000:0"
  for item in items:
    if item.get("delay_schedule") != expected_delay:
      errors.append(f"{item.get('run_id')} does not use the frozen delay schedule")
    if item["controller"] == "adaptive" and int(item["initial_horizon"]) != 8:
      errors.append(f"{item['run_id']} must start adaptive search at h=8")
    if item["controller"] in {"scripted", "fixed"} and int(
        item["initial_horizon"]
    ) != 3:
      errors.append(f"{item['run_id']} must start at h=3")
    if item["controller"] == "scripted" and item.get(
        "horizon_schedule"
    ) != "0:3,150000:7,350000:3":
      errors.append(f"{item['run_id']} has the wrong scripted schedule")
  calibration_profiles = [
      item for item in items
      if item.get("probe_timing") or item.get("reference_probe")
  ]
  if (
      len(calibration_profiles) != 1 or
      calibration_profiles[0].get("probe_timing") is not True or
      calibration_profiles[0].get("reference_probe") is not True or
      calibration_profiles[0].get("run_id") != "cpdelay__additive__s1"
  ):
    errors.append("additive seed 1 must be the sole timing/reference profile")
  fixed = [item for item in items if item["controller"] == "fixed"]
  if (
      len(fixed) != 1 or
      fixed[0].get("delay_schedule") != expected_delay or
      fixed[0].get("augment_delay_observation") is not True
  ):
    errors.append("exactly one matched fixed-h3 delay control is required")
  if not (ROOT / "scripts/ncc_cartpole_delay_pilot.sbatch").exists():
    errors.append("pilot sbatch script is missing")
  if not ledger_path(goal).exists():
    errors.append("pilot ledger is missing")
  return errors


def launch_blockers(goal: dict[str, Any]) -> list[str]:
  blockers = validate(goal)
  local_commit = git(ROOT, "rev-parse", "HEAD")
  dirty = scientific_dirty_paths()
  if goal["constraints"].get("require_clean_local_git") and dirty:
    blockers.append(f"local scientific files are dirty: {dirty}")
  gates = gate_state(goal)
  if not gates.get("full_launch_ready"):
    blockers.append("full-launch gate is not ready")
  for required_gate in goal["gates"].get("required", ()): 
    if gates.get(required_gate) is not True:
      blockers.append(f"required gate {required_gate!r} has not passed")
  if gates.get("git_commit") != local_commit:
    blockers.append("gate revision does not match local HEAD")
  try:
    remote_state = remote_repo_state(goal)
  except Exception as exc:
    blockers.append(f"remote pilot checkout unavailable: {exc}")
    return blockers
  if goal["constraints"].get("require_clean_remote_git") and remote_state.get("dirty") != "0":
    blockers.append("remote pilot checkout is dirty")
  if (
      goal["constraints"].get("require_remote_commit_match")
      and remote_state.get("commit") != local_commit
  ):
    blockers.append("remote pilot revision does not match local HEAD")
  if remote_state.get("branch") != goal["repo"]["branch"]:
    blockers.append("remote pilot checkout is on the wrong branch")
  return blockers


def base_row(goal: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
  return {
      "run_id": profile["run_id"],
      "controller": profile["controller"],
      "score_mode": profile["score_mode"],
      "seed": profile["seed"],
      "git_commit": git(ROOT, "rev-parse", "HEAD"),
      "config_hash": profile_hash(goal, profile),
      "remote_run_dir": (
          f"{goal['tracking']['remote_run_dir_prefix']}/{profile['run_id']}"
      ),
  }


def validate_remote_run(
    goal: dict[str, Any],
    profile: dict[str, Any],
    remote_run_dir: str,
    *,
    expected_commit: str,
    expected_config_hash: str,
) -> tuple[bool, str]:
  repo = shlex.quote(goal["repo"]["remote_path"])
  run_dir = shlex.quote(f"{goal['repo']['remote_path']}/{remote_run_dir}")
  command = (
      f"cd {repo} && {REMOTE_PYTHON} scripts/validate_cartpole_delay_run.py "
      f"--run-dir {run_dir} "
      f"--controller {shlex.quote(profile['controller'])} "
      f"--expected-run-id {shlex.quote(profile['run_id'])} "
      f"--expected-score-mode {shlex.quote(profile['score_mode'])} "
      f"--expected-seed {int(profile['seed'])} "
      f"--expected-commit {shlex.quote(expected_commit)} "
      f"--expected-config-hash {shlex.quote(expected_config_hash)}"
  )
  if profile.get("probe_timing", False):
    command += " --require-probe-timing"
  if profile.get("reference_probe", False):
    command += " --require-reference-probe"
  # Artifact-invalid is a scientific outcome, not an SSH transport failure.
  # Keep the remote shell successful so the controller can durably ledger it.
  result = remote(command + " 2>&1 || true")
  lines = result.splitlines()
  return bool(lines and lines[-1].strip() == "VALID"), result


def read_remote_manifest(goal: dict[str, Any], remote_run_dir: str) -> dict[str, Any]:
  manifest = shlex.quote(
      f"{goal['repo']['remote_path']}/{remote_run_dir}/artifacts/run_manifest.json"
  )
  return json.loads(remote(f"cat {manifest}"))


def process_terminal_jobs(
    goal: dict[str, Any], rows: list[dict[str, str]], *, dry_run: bool
) -> list[str]:
  messages: list[str] = []
  by_id = {item["run_id"]: item for item in profiles(goal)}
  for run_id, row in open_jobs(rows).items():
    state, exit_code, elapsed = slurm_state(row["job_id"])
    if state not in TERMINAL_STATES:
      messages.append(f"active {run_id} job={row['job_id']} state={state} elapsed={elapsed}")
      continue
    profile = by_id[run_id]
    event = base_row(goal, profile)
    event.update(
        {
            "job_id": row["job_id"],
            "attempt": row["attempt"],
            "remote_run_dir": row["remote_run_dir"],
            "slurm_state": state,
            "exit_code": exit_code,
            "git_commit": row["git_commit"],
            "config_hash": row["config_hash"],
        }
    )
    if state == "COMPLETED" and exit_code.startswith("0:"):
      valid, detail = validate_remote_run(
          goal,
          profile,
          row["remote_run_dir"],
          expected_commit=row["git_commit"],
          expected_config_hash=row["config_hash"],
      )
      if valid:
        manifest = read_remote_manifest(goal, row["remote_run_dir"])
        event.update(
            {
                "event_type": "completed",
                "status": "valid",
                "gifs_valid": "true",
                "probe_artifacts_valid": (
                    "true" if profile["controller"] == "adaptive" else "not_applicable"
                ),
                "final_eval_mean": manifest.get("final_eval_mean", ""),
                "final_eval_std": manifest.get("final_eval_std", ""),
                "notes": f"validated; elapsed={elapsed}",
            }
        )
      else:
        event.update(
            {
                "event_type": "failed",
                "status": "invalid_artifacts",
                "failure_class": "deterministic_artifact_validation",
                "notes": detail[-1000:],
            }
        )
    else:
      failure_class = (
          "transient_infrastructure"
          if state in TRANSIENT_STATES
          else "resource_oom"
          if state == "OUT_OF_MEMORY"
          else "timeout_requires_diagnosis"
          if state == "TIMEOUT"
          else "deterministic_requires_diagnosis"
      )
      event.update(
          {
              "event_type": "failed",
              "status": "failed",
              "failure_class": failure_class,
              "notes": f"terminal Slurm state; elapsed={elapsed}",
          }
      )
    messages.append(
        f"terminal {run_id} job={row['job_id']} state={state} -> {event['event_type']}"
    )
    if not dry_run:
      append_ledger(goal, event)
  return messages


def retry_allowed(
    goal: dict[str, Any], rows: list[dict[str, str]], profile: dict[str, Any]
) -> tuple[bool, str]:
  count = attempts_for(rows, profile["run_id"])
  max_attempts = 1 + int(goal["constraints"]["max_retries"])
  if count >= max_attempts:
    return False, f"retry limit reached ({count}/{max_attempts})"
  latest = latest_event(rows, profile["run_id"])
  if latest is None or latest.get("event_type") != "failed":
    return True, "never launched"
  if latest.get("failure_class") == "transient_infrastructure":
    return True, "transient infrastructure retry"
  current_commit = git(ROOT, "rev-parse", "HEAD")
  if latest.get("git_commit") and latest.get("git_commit") != current_commit:
    return True, "new tested revision after diagnosed failure"
  return False, "scientific/resource failure requires diagnosis and a new tested revision"


def launch_profile(
    goal: dict[str, Any], profile: dict[str, Any], rows: list[dict[str, str]], *, dry_run: bool
) -> str:
  attempt = attempts_for(rows, profile["run_id"]) + 1
  remote_repo = goal["repo"]["remote_path"]
  run_dir = (
      f"{goal['tracking']['remote_run_dir_prefix']}/{profile['run_id']}/"
      f"attempt_{attempt}"
  )
  exports = {
      "RUN_ID": profile["run_id"],
      "CONTROLLER": profile["controller"],
      "SCORE_MODE": profile["score_mode"],
      "SEED": profile["seed"],
      "INITIAL_HORIZON": profile["initial_horizon"],
      "AUGMENT_DELAY_OBSERVATION": str(
          profile.get("augment_delay_observation", True)
      ).lower(),
      "PROBE_TIMING": str(profile.get("probe_timing", False)).lower(),
      "REFERENCE_PROBE": str(profile.get("reference_probe", False)).lower(),
      "RUN_DIR": run_dir,
      "ATTEMPT": attempt,
      "EXPECTED_COMMIT": git(ROOT, "rev-parse", "HEAD"),
      "CONFIG_HASH": profile_hash(goal, profile),
  }
  export_text = ",".join(f"{key}={value}" for key, value in exports.items())
  job_name = profile_job_name(profile, attempt)
  command = (
      f"cd {shlex.quote(remote_repo)} && "
      f"sbatch --parsable --job-name={shlex.quote(job_name)} "
      f"--export={shlex.quote('ALL,' + export_text)} "
      "scripts/ncc_cartpole_delay_pilot.sbatch"
  )
  if dry_run:
    return f"would launch {profile['run_id']} attempt={attempt}: {command}"
  existing = remote(
      f"squeue -h -u goncalo -n {shlex.quote(job_name)} -o '%A|%T'"
  ).splitlines()
  if not existing:
    accounting = remote(
        "sacct -X -n "
        f"--name {shlex.quote(job_name)} "
        "--format=JobIDRaw,State -P"
    ).splitlines()
    existing = [
        line for line in accounting
        if re.fullmatch(r"\d+\|.+", line.strip())
    ][-1:]
  if existing:
    job_id, state = (existing[0].split("|", 1) + [""])[:2]
    if not re.fullmatch(r"\d+", job_id.strip()):
      raise RuntimeError(f"could not reconcile existing job: {existing[0]!r}")
    row = base_row(goal, profile)
    row.update(
        {
            "event_type": "launched",
            "job_id": job_id.strip(),
            "attempt": attempt,
            "status": "reconciled",
            "slurm_state": state.strip(),
            "remote_run_dir": run_dir,
            "notes": f"reconciled pre-existing job_name={job_name}",
        }
    )
    append_ledger(goal, row)
    return f"reconciled {profile['run_id']} attempt={attempt} job={job_id.strip()}"
  intent = base_row(goal, profile)
  intent.update(
      {
          "event_type": "launch_intent",
          "attempt": attempt,
          "status": "submitting",
          "remote_run_dir": run_dir,
          "notes": f"job_name={job_name}",
      }
  )
  append_ledger(goal, intent)
  # Submission is non-idempotent. A lost SSH response is reconciled by the
  # durable launch_intent and attempt-specific job name on the next tick.
  output = remote(command, attempts=1)
  job_id = output.splitlines()[-1].split(";", 1)[0].strip()
  if not re.fullmatch(r"\d+", job_id):
    raise RuntimeError(f"could not parse sbatch job id from: {output!r}")
  row = base_row(goal, profile)
  row.update(
      {
          "event_type": "launched",
          "job_id": job_id,
          "attempt": attempt,
          "status": "submitted",
          "slurm_state": "SUBMITTED",
          "remote_run_dir": run_dir,
          "notes": f"job_name={job_name}",
      }
  )
  append_ledger(goal, row)
  return f"launched {profile['run_id']} attempt={attempt} job={job_id}"


def profile_job_name(profile: dict[str, Any], attempt: int) -> str:
  """Returns an attempt-specific Slurm identity for safe reconciliation."""
  base = profile["run_id"].replace("cpdelay__", "cpdelay-").replace("__", "-")
  return f"{base[:88]}-a{int(attempt)}"


def status_report(goal: dict[str, Any], *, include_remote: bool = True) -> str:
  rows = read_ledger(goal)
  completed = completed_ids(rows)
  open_map = open_jobs(rows)
  pending = [item["run_id"] for item in profiles(goal) if item["run_id"] not in completed and item["run_id"] not in open_map]
  lines = [
      f"goal={goal['name']}",
      f"local_commit={git(ROOT, 'rev-parse', 'HEAD')}",
      f"local_scientific_dirty={scientific_dirty_paths()}",
      f"completed={len(completed)}/{len(profiles(goal))}",
      f"open_jobs={len(open_map)}",
      f"pending_profiles={len(pending)}",
      f"ledger={goal['tracking']['ledger']}",
      f"gates={json.dumps(gate_state(goal), sort_keys=True)}",
  ]
  if include_remote:
    try:
      gpu_raw, _, free = gpu_snapshot(goal)
      lines.append(f"physically_free_gpus={free}")
      lines.append("gpu_rows:\n" + gpu_raw)
      lines.append("remote_repo=" + json.dumps(remote_repo_state(goal), sort_keys=True))
      lines.append(
          "queue:\n"
          + remote("squeue -u goncalo -o '%.18i %.9P %.30j %.2t %.10M %.10l %R'")
      )
    except Exception as exc:
      lines.append(f"remote_error={exc}")
  for run_id, row in open_map.items():
    lines.append(f"open {run_id} job={row['job_id']} attempt={row['attempt']}")
  for run_id in pending:
    lines.append(f"pending {run_id}")
  return "\n".join(lines)


def tick(goal: dict[str, Any], *, dry_run: bool) -> str:
  errors = validate(goal)
  if errors:
    return "validation_failed:\n- " + "\n- ".join(errors)
  rows = read_ledger(goal)
  messages = process_terminal_jobs(goal, rows, dry_run=dry_run)
  if not dry_run:
    rows = read_ledger(goal)
  blockers = launch_blockers(goal)
  if blockers:
    messages.append("launch_blocked:\n- " + "\n- ".join(blockers))
    messages.append(status_report(goal))
    return "\n".join(messages)
  _, _, physically_free = gpu_snapshot(goal)
  active = len(open_jobs(rows))
  capacity = max(
      0,
      min(
          int(goal["constraints"]["max_active_gpus"]) - active,
          physically_free,
      ),
  )
  completed = completed_ids(rows)
  open_ids = set(open_jobs(rows))
  for profile in profiles(goal):
    if capacity <= 0:
      break
    if profile["run_id"] in completed or profile["run_id"] in open_ids:
      continue
    allowed, reason = retry_allowed(goal, rows, profile)
    if not allowed:
      messages.append(f"blocked {profile['run_id']}: {reason}")
      continue
    messages.append(launch_profile(goal, profile, rows, dry_run=dry_run))
    capacity -= 1
    if not dry_run:
      rows = read_ledger(goal)
  messages.append(status_report(goal))
  return "\n".join(messages)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--goal", type=Path, default=DEFAULT_GOAL)
  sub = parser.add_subparsers(dest="command", required=True)
  sub.add_parser("validate")
  sub.add_parser("status")
  tick_parser = sub.add_parser("tick")
  tick_parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  goal = load_goal(args.goal)
  if args.command == "validate":
    errors = validate(goal)
    if errors:
      print("INVALID\n- " + "\n- ".join(errors))
      raise SystemExit(2)
    print(f"VALID profiles={len(profiles(goal))}")
  elif args.command == "status":
    print(status_report(goal))
  elif args.command == "tick":
    lock_path = ROOT / "runs/results/cartpole_delay_pilot/.steward.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
      try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError:
        print("tick_skipped: another Cartpole delay steward owns the lock")
        return
      print(tick(goal, dry_run=bool(args.dry_run)))


if __name__ == "__main__":
  main()
