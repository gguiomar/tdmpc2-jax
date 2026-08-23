import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _load_analysis_module():
  path = Path(__file__).resolve().parents[1] / 'scripts' / 'finalize_cartpole_delay_run.py'
  spec = importlib.util.spec_from_file_location('cartpole_delay_analysis', path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class DecisionReconstructionTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.analysis = _load_analysis_module()

  def test_incumbent_has_zero_score_and_se(self):
    returns = {
        2: [1.0, 2.0, 3.0, 4.0],
        3: [2.0, 3.0, 4.0, 5.0],
        4: [3.0, 4.0, 5.0, 6.0],
    }
    projections = {
        2: [1.0, -1.0, 1.0, -1.0],
        3: [2.0, -2.0, 2.0, -2.0],
        4: [3.0, -3.0, 3.0, -3.0],
    }
    result = self.analysis.reconstruct_decision(
        horizons=[2, 3, 4],
        incumbent=3,
        paired_returns=returns,
        projections=projections,
        score_mode='additive',
        probe_count=4,
    )
    self.assertIsNotNone(result)
    self.assertAlmostEqual(float(result['score'][1]), 0.0)
    self.assertAlmostEqual(float(result['score_se'][1]), 0.0)

  def test_roughness_matches_jax_epsilon_convention(self):
    returns = {2: [1.0, 1.1], 3: [1.0, 1.1]}
    projections = {2: [0.0, 0.0], 3: [0.0, 0.0]}
    result = self.analysis.reconstruct_decision(
        horizons=[2, 3],
        incumbent=2,
        paired_returns=returns,
        projections=projections,
        score_mode='multiplicative',
        probe_count=2,
    )
    np.testing.assert_allclose(result['roughness'], np.full(2, 1e-3))

  def test_return_artifact_deduplicates_and_preserves_three_sources(self):
    rows = [
        {'step': '20', 'tag': 'dense_rhs/candidate_3_return_replica_0', 'value': '1'},
        {'step': '20', 'tag': 'dense_rhs/candidate_3_return_replica_0', 'value': '2'},
        {
            'step': '20',
            'tag': 'reference_probe/dense_rhs/candidate_3_return_replica_0',
            'value': '3',
        },
        {
            'step': '20',
            'tag': (
                'conditional_reference_probe/dense_rhs/'
                'candidate_3_return_replica_0'
            ),
            'value': '4',
        },
    ]
    with tempfile.TemporaryDirectory() as temporary_dir:
      metrics_dir = Path(temporary_dir)
      record_count, _ = self.analysis.build_return_artifacts(metrics_dir, rows)
      with np.load(metrics_dir / 'paired_returns.npz', allow_pickle=False) as data:
        sources = set(np.asarray(data['source']).astype(str).tolist())
        deployed = np.asarray(data['episode_return'])[
            np.asarray(data['source']).astype(str) == 'deployed'
        ]

    self.assertEqual(record_count, 3)
    self.assertEqual(
        sources,
        {'deployed', 'reference', 'conditional_reference'},
    )
    np.testing.assert_array_equal(deployed, [2.0])

  def test_reference_probe_uses_explicit_shadow_incumbent(self):
    rows = [
        {
            'step': '100000',
            'tag': 'reference_probe/dense_rhs/candidate_4_roughness_m2',
            'value': '2.0',
        },
        {
            'step': '100000',
            'tag': 'reference_probe/dense_rhs/candidate_4_roughness_m64',
            'value': '1.0',
        },
        {
            'step': '100000',
            'tag': 'reference_probe/dense_rhs/selected_horizon',
            'value': '3',
        },
        {
            'step': '100000',
            'tag': 'reference_probe/dense_rhs/proposed_horizon',
            'value': '3',
        },
        {
            'step': '100000',
            'tag': 'reference_probe/incumbent_horizon',
            'value': '4',
        },
        {
            'step': '100000',
            'tag': 'dense_rhs/previous_horizon',
            'value': '2',
        },
    ]
    with tempfile.TemporaryDirectory() as temporary_dir:
      metrics_dir = Path(temporary_dir)
      self.analysis.build_probe_calibration(metrics_dir, rows)
      with (metrics_dir / 'probe_calibration.csv').open(newline='') as handle:
        output = list(csv.DictReader(handle))

    self.assertTrue(output)
    self.assertTrue(all(row['previous_horizon'] == '4.0' for row in output))
    self.assertTrue(all(row['switch'] == '1' for row in output))

  def test_roughness_bootstrap_reports_all_nested_probe_counts(self):
    rows = [
        {
            'step': '20000',
            'tag': f'dense_rhs/horizon_3_roughness_projection_{direction}',
            'value': str(direction + 1),
        }
        for direction in range(64)
    ]
    with tempfile.TemporaryDirectory() as temporary_dir:
      metrics_dir = Path(temporary_dir)
      count = self.analysis.build_roughness_bootstrap(
          metrics_dir,
          rows,
          bootstrap_replicates=100,
      )
      with (metrics_dir / 'roughness_bootstrap.csv').open(newline='') as handle:
        output = list(csv.DictReader(handle))

    self.assertEqual(count, 6)
    self.assertEqual(
        {int(row['probe_count']) for row in output},
        {2, 4, 8, 16, 32, 64},
    )
    self.assertTrue(all(float(row['bootstrap_se']) >= 0.0 for row in output))


if __name__ == '__main__':
  unittest.main()
