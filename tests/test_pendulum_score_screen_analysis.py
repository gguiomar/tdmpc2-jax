from scripts.analyze_pendulum_score_screen import _full_run_candidates


def test_full_run_candidates_match_frozen_profile_run_ids(tmp_path):
  expected = (
      tmp_path / 'pendscore__s0_current_additive__s7' / 'attempt_1'
  )
  expected.mkdir(parents=True)
  (tmp_path / 'pendscore__smoke_s3__s7' / 'attempt_1').mkdir(parents=True)

  assert _full_run_candidates(tmp_path, 's0') == [expected]
