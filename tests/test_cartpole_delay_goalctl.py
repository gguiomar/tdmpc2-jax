import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


def _load_goalctl():
  path = Path(__file__).resolve().parents[1] / 'scripts' / 'cartpole_delay_goalctl.py'
  spec = importlib.util.spec_from_file_location('cartpole_delay_goalctl', path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class StewardIdentityTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.goalctl = _load_goalctl()

  def test_scientific_dirty_parser_preserves_first_porcelain_column(self):
    status = (
        ' M experiments/cartpole_delay_pilot_ledger.csv\n'
        ' M tdmpc2_jax/train.py\n'
    )
    completed = SimpleNamespace(stdout=status)
    with mock.patch.object(self.goalctl, 'run_local', return_value=completed):
      self.assertEqual(
          self.goalctl.scientific_dirty_paths(),
          ['tdmpc2_jax/train.py'],
      )

  def test_slurm_cancellation_reasons_normalize_to_terminal_state(self):
    self.assertEqual(
        self.goalctl.normalize_slurm_state('CANCELLED by 12345'),
        'CANCELLED',
    )
    self.assertEqual(self.goalctl.normalize_slurm_state('COMPLETED+'), 'COMPLETED')

  def test_job_names_are_attempt_specific(self):
    profile = {'run_id': 'cpdelay__additive__s1'}
    self.assertEqual(
        self.goalctl.profile_job_name(profile, 1),
        'cpdelay-additive-s1-a1',
    )
    self.assertNotEqual(
        self.goalctl.profile_job_name(profile, 1),
        self.goalctl.profile_job_name(profile, 2),
    )

  def test_remote_validation_uses_launched_identity(self):
    goal = {'repo': {'remote_path': '/remote/pilot'}}
    profile = {
        'controller': 'adaptive',
        'run_id': 'cpdelay__additive__s1',
        'score_mode': 'additive',
        'seed': 1,
    }
    captured = []

    def fake_remote(command):
      captured.append(command)
      return 'VALID'

    with mock.patch.object(self.goalctl, 'remote', side_effect=fake_remote):
      valid, _ = self.goalctl.validate_remote_run(
          goal,
          profile,
          'outputs/run/attempt_1',
          expected_commit='launched-commit',
          expected_config_hash='launched-config',
      )

    self.assertTrue(valid)
    self.assertIn('--expected-commit launched-commit', captured[0])
    self.assertIn('--expected-config-hash launched-config', captured[0])


if __name__ == '__main__':
  unittest.main()
