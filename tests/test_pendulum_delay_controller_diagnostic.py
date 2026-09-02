import hashlib
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'analyze_pendulum_delay_controller_diagnostic.py'
)
SPEC = importlib.util.spec_from_file_location('pendulum_delay_diagnostic', MODULE_PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnostic)


def test_forced_horizon_contracts_follow_delay_and_causal_coverage():
  expected_b3 = {34_000: 2, 38_000: 2, 42_000: 6, 46_000: 4, 50_000: 2}
  expected_b4 = {34_000: 2, 38_000: 3, 42_000: 7, 46_000: 5, 50_000: 2}

  for step, horizon in expected_b3.items():
    assert diagnostic._expected_horizon('b3', 'full', step) == horizon
  for step, horizon in expected_b4.items():
    assert diagnostic._expected_horizon('b4', 'full', step) == horizon


def test_config_hash_matches_the_launcher_projection():
  commit = '0123456789abcdef'
  payload = (
      'pendulum-delay-controller-diagnostic-v1'
      f'|commit={commit}|mode=full|profile=b5'
      '|run=penddiag__b5_return_argmax__s7'
      '|source=34000|final=54000|seed=7'
  )
  expected = hashlib.sha256(payload.encode('utf-8')).hexdigest()

  assert diagnostic._expected_config_hash(
      commit,
      'full',
      'b5',
      'penddiag__b5_return_argmax__s7',
      54_000,
  ) == expected


def test_smoke_and_full_run_ids_are_frozen():
  assert diagnostic._expected_run_id('smoke', 'b4') == 'penddiag__smoke_b4__s7'
  assert diagnostic._expected_run_id('full', 'b2') == 'penddiag__b2_fixed_h3__s7'

