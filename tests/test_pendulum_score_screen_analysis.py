import numpy as np

from scripts.analyze_pendulum_score_screen import (
    _confirmation_profiles,
    _full_run_candidates,
    _trapezoidal_mean,
)


def test_full_run_candidates_match_frozen_profile_run_ids(tmp_path):
  expected = (
      tmp_path / 'pendscore__s0_current_additive__s7' / 'attempt_1'
  )
  expected.mkdir(parents=True)
  (tmp_path / 'pendscore__smoke_s3__s7' / 'attempt_1').mkdir(parents=True)

  assert _full_run_candidates(tmp_path, 's0') == [expected]


def test_trapezoidal_mean_uses_current_numpy_api():
  steps = np.asarray([0.0, 1.0, 2.0])
  values = np.asarray([0.0, 1.0, 2.0])

  assert _trapezoidal_mean(values, steps) == 1.0


def test_confirmation_profiles_exclude_ineligible_arms():
  runs = {
      's0': {'metrics': {'mean_oracle_regret': 1.6}},
      's1': {'metrics': {'mean_oracle_regret': 3.3}},
      's2': {'metrics': {'mean_oracle_regret': 1.3}},
  }

  assert _confirmation_profiles(runs, ['s0']) == ['s0']
