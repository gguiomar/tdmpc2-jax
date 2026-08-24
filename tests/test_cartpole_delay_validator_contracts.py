from scripts.validate_cartpole_delay_run import contract_expectations


def test_stabilization_contract_is_not_treated_as_smoke():
  values = contract_expectations('stabilization', 50_000)
  assert values['save_interval_steps'] == 50_000
  assert values['evaluation_interval_steps'] == 2_500
  assert values['evaluation_num_episodes'] == 10
  assert values['action_delay_schedule_enabled'] is False
  assert values['calibration_anchors'] == [100_000, 250_000, 450_000]


def test_auto_contract_preserves_legacy_smoke_behavior():
  values = contract_expectations('auto', 24_000)
  assert values['name'] == 'smoke'
  assert values['evaluation_interval_steps'] == 24_000
  assert values['timing_repetitions'] == 2


def test_auto_contract_preserves_full_behavior():
  values = contract_expectations('auto', 500_000)
  assert values['name'] == 'full'
  assert values['evaluation_interval_steps'] == 50_000
  assert values['timing_repetitions'] == 30
