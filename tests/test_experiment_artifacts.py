import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_artifact_helpers():
  """Load pure artifact helpers without importing optional training packages."""
  source_path = REPO_ROOT / 'tdmpc2_jax' / 'train.py'
  tree = ast.parse(source_path.read_text())
  selected = [
      node
      for node in tree.body
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
      node.name in {
          '_open_csv_for_append',
          '_next_episode_indices',
          '_scripted_horizon_schedule',
          '_scripted_horizon_at_step',
          '_next_scripted_horizon_step',
          '_validate_dense_rhs_deployment_mode',
          '_resolve_dense_query_deployment',
          '_reindex_shadow_metrics_at_deployment',
          '_restored_dense_query_interval',
          '_restored_dense_query_step',
          '_select_horizon_state_on_restore',
          '_dense_query_due_at_boundary',
          '_make_full_horizon_deployed_planner_agent',
      }
  ]
  namespace = {
      'csv': csv,
      'os': os,
      'Path': Path,
      'np': np,
      'TDMPC2': object,
  }
  exec(compile(ast.Module(body=selected, type_ignores=[]), source_path, 'exec'), namespace)
  return namespace


def _load_render_module():
  script_path = REPO_ROOT / 'scripts' / 'render_cartpole_delay_gifs.py'
  spec = importlib.util.spec_from_file_location('render_cartpole_delay_gifs', script_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class AppendSafeCsvTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.helpers = _load_train_artifact_helpers()

  def test_resume_appends_without_duplicate_header_or_truncation(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      path = Path(temporary_dir) / 'metrics.csv'
      for value in ('first', 'second'):
        output_file, writer = self.helpers['_open_csv_for_append'](
            path,
            ['step', 'tag'],
        )
        writer.writerow({'step': 1, 'tag': value})
        output_file.close()

      with path.open(newline='') as input_file:
        rows = list(csv.reader(input_file))
      self.assertEqual(rows[0], ['step', 'tag'])
      self.assertEqual(rows.count(['step', 'tag']), 1)
      self.assertEqual(rows[1:], [['1', 'first'], ['1', 'second']])

  def test_resume_repairs_missing_final_newline(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      path = Path(temporary_dir) / 'metrics.csv'
      path.write_bytes(b'step,tag\n1,first')
      output_file, writer = self.helpers['_open_csv_for_append'](
          path,
          ['step', 'tag'],
      )
      writer.writerow({'step': 2, 'tag': 'second'})
      output_file.close()
      self.assertEqual(
          path.read_text().splitlines(),
          ['step,tag', '1,first', '2,second'],
      )

  def test_schema_mismatch_fails_before_append(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      path = Path(temporary_dir) / 'metrics.csv'
      path.write_text('wrong,header\n')
      with self.assertRaises(ValueError):
        self.helpers['_open_csv_for_append'](path, ['step', 'tag'])
      self.assertEqual(path.read_text(), 'wrong,header\n')

  def test_episode_indices_continue_per_environment(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      path = Path(temporary_dir) / 'episodes.csv'
      path.write_text(
          'env_index,episode_index\n'
          '0,3\n'
          '1,8\n'
          '0,5\n'
      )
      actual = self.helpers['_next_episode_indices'](path, 3)
      np.testing.assert_array_equal(actual, [6, 9, 0])


class ScriptedHorizonTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.helpers = _load_train_artifact_helpers()

  def test_three_seven_three_boundaries_are_exact(self):
    config = {
        'scripted_horizon': {
            'enabled': True,
            'schedule_steps': [0, 150_000, 350_000],
            'schedule_values': [3, 7, 3],
        }
    }
    schedule = self.helpers['_scripted_horizon_schedule'](config)
    at_step = self.helpers['_scripted_horizon_at_step']
    self.assertEqual(at_step(schedule, 0, 99), 3)
    self.assertEqual(at_step(schedule, 149_999, 99), 3)
    self.assertEqual(at_step(schedule, 150_000, 99), 7)
    self.assertEqual(at_step(schedule, 349_999, 99), 7)
    self.assertEqual(at_step(schedule, 350_000, 99), 3)

  def test_next_boundary_is_strictly_after_current_step(self):
    schedule = ((0, 3), (150_000, 7), (350_000, 3))
    next_step = self.helpers['_next_scripted_horizon_step']
    self.assertEqual(next_step(schedule, 0, 999_999), 150_000)
    self.assertEqual(next_step(schedule, 150_000, 999_999), 350_000)
    self.assertEqual(next_step(schedule, 350_000, 999_999), 999_999)

  def test_invalid_schedule_is_rejected(self):
    config = {
        'scripted_horizon': {
            'enabled': True,
            'schedule_steps': [0, 150_000, 150_000],
            'schedule_values': [3, 7, 3],
        }
    }
    with self.assertRaises(ValueError):
      self.helpers['_scripted_horizon_schedule'](config)

  def test_scripted_and_dense_require_explicit_shadow_only_mode(self):
    validate = self.helpers['_validate_dense_rhs_deployment_mode']
    schedule = ((0, 2), (42_000, 6))
    with self.assertRaisesRegex(ValueError, 'shadow_only=true'):
      validate(
          scripted_schedule=schedule,
          dense_rhs_enabled=True,
          shadow_only=False,
          env_backend='mjx_dmc',
      )
    validate(
        scripted_schedule=schedule,
        dense_rhs_enabled=True,
        shadow_only=True,
        env_backend='mjx_dmc',
    )

  def test_shadow_only_script_is_rejected_outside_mjx_loop(self):
    with self.assertRaisesRegex(ValueError, 'only by the MJX-DMC'):
      self.helpers['_validate_dense_rhs_deployment_mode'](
          scripted_schedule=((0, 3),),
          dense_rhs_enabled=True,
          shadow_only=True,
          env_backend='dmc',
      )

  def test_shadow_query_cannot_change_deployed_horizon(self):
    resolve = self.helpers['_resolve_dense_query_deployment']
    self.assertEqual(resolve(3, 8, True), (3, 8))
    self.assertEqual(resolve(3, 8, False), (8, 8))

  def test_shadow_query_tracks_scripted_boundaries_without_owning_deployment(self):
    at_step = self.helpers['_scripted_horizon_at_step']
    resolve = self.helpers['_resolve_dense_query_deployment']
    schedule = (
        (0, 2),
        (38_000, 3),
        (42_000, 7),
        (46_000, 5),
        (50_000, 2),
    )

    for step, adaptive, expected_deployed in (
        (37_999, 8, 2),
        (38_000, 8, 3),
        (42_000, 2, 7),
        (46_000, 8, 5),
        (50_000, 8, 2),
    ):
      scripted = at_step(schedule, step, 99)
      self.assertEqual(
          resolve(scripted, adaptive, True),
          (expected_deployed, adaptive),
      )

  def test_shadow_best_aliases_are_reindexed_to_deployment(self):
    reindex = self.helpers['_reindex_shadow_metrics_at_deployment']
    actual = reindex({
        'dense_rhs/best_fitness': 8.0,
        'dense_rhs/deployment_score_best': 80.0,
        'dense_rhs/candidate_3_fitness': 3.0,
        'dense_rhs/candidate_3_deployment_score': 30.0,
    }, 3)
    self.assertEqual(actual['dense_rhs/best_fitness'], 3.0)
    self.assertEqual(actual['dense_rhs/deployment_score_best'], 30.0)
    self.assertEqual(actual['dense_rhs/shadow_controller_best_fitness'], 8.0)
    self.assertEqual(
        actual['dense_rhs/shadow_controller_deployment_score_best'],
        80.0,
    )

  def test_conditional_reference_expands_plan_buffer_to_search_hmax(self):
    class FakeAgent:
      def __init__(self, **values):
        self.values = values

      def replace(self, **updates):
        return FakeAgent(**(self.values | updates))

    agent = FakeAgent(
        horizon=3,
        planning_hmax=3,
        population_size=512,
        policy_prior_samples=24,
        num_elites=64,
        mppi_iterations=6,
        temperature=0.5,
    )
    expanded = self.helpers['_make_full_horizon_deployed_planner_agent'](
        agent,
        8,
    )

    self.assertEqual(expanded.values['planning_hmax'], 8)
    self.assertEqual(expanded.values['horizon'], 3)
    self.assertEqual(expanded.values['population_size'], 512)
    self.assertEqual(expanded.values['mppi_iterations'], 6)

  def test_checkpoint_fork_can_reset_query_schedule(self):
    helper = self.helpers['_restored_dense_query_step']
    self.assertEqual(
        helper(
            30_000,
            {
                'reset_query_schedule_on_restore': True,
                'start_query_step': 32_000,
            },
            34_000,
        ),
        32_000,
    )

    self.assertEqual(
        helper(
            30_000,
            {
                'reset_query_schedule_on_restore': False,
                'start_query_step': 32_000,
            },
            34_000,
        ),
        34_000,
    )

    interval = self.helpers['_restored_dense_query_interval']
    self.assertEqual(
        interval(
            {
                'reset_query_schedule_on_restore': True,
                'query_interval_steps': 400,
            },
            4_000,
        ),
        400,
    )
    self.assertEqual(
        interval(
            {
                'reset_query_schedule_on_restore': False,
                'query_interval_steps': 400,
            },
            4_000,
        ),
        4_000,
    )

  def test_checkpoint_fork_can_reset_all_controller_evidence(self):
    select = self.helpers['_select_horizon_state_on_restore']
    fresh = object()
    inherited = object()
    self.assertIs(select(fresh, inherited, True), fresh)
    self.assertIs(select(fresh, inherited, False), inherited)

  def test_dense_query_includes_exact_terminal_boundary(self):
    class QueryState:
      def __init__(self, next_query_step):
        self.next_query_step = next_query_step

      def should_query(self, step):
        return int(step) >= int(self.next_query_step)

    due = self.helpers['_dense_query_due_at_boundary']
    state = QueryState(36_000)
    self.assertFalse(due(35_600, 36_000, state))
    self.assertTrue(due(36_000, 36_000, state))
    self.assertFalse(due(36_008, 36_000, state))


class RolloutValidationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.render = _load_render_module()

  @staticmethod
  def _write_trajectory(path: Path):
    frames = 4
    action_steps = frames - 1
    initial_states = 2
    np.savez_compressed(
        path,
        qpos=np.zeros((frames, initial_states, 2), dtype=np.float32),
        qvel=np.zeros((frames, initial_states, 2), dtype=np.float32),
        ctrl=np.zeros((frames, initial_states, 1), dtype=np.float32),
        commanded_action=np.zeros(
            (action_steps, initial_states, 1), dtype=np.float32
        ),
        applied_action=np.zeros(
            (action_steps, initial_states, 1), dtype=np.float32
        ),
        delayed_actions=np.zeros(
            (action_steps, initial_states, 4, 1), dtype=np.float32
        ),
        reward=np.zeros((action_steps, initial_states), dtype=np.float32),
        done=np.zeros((action_steps, initial_states), dtype=bool),
        effective_action_delay=np.zeros(
            (frames, initial_states), dtype=np.int32
        ),
        frame_timestamp_seconds=np.arange(frames, dtype=np.float64) * 0.02,
    )

  def test_standard_rollout_layout_validates(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      rollout_dir = Path(temporary_dir) / 'step_100000'
      rollout_dir.mkdir()
      metadata = {
          'global_step': 100_000,
          'selected_horizon': 3,
          'environment': {'task': 'cartpole-swingup'},
      }
      (rollout_dir / 'metadata.json').write_text(json.dumps(metadata))
      self._write_trajectory(rollout_dir / 'trajectory_delay0.npz')
      self._write_trajectory(rollout_dir / 'trajectory_delay4.npz')
      self.assertEqual(
          self.render.validate_rollout_dir(rollout_dir),
          metadata,
      )

  def test_missing_required_array_is_rejected(self):
    with tempfile.TemporaryDirectory() as temporary_dir:
      rollout_dir = Path(temporary_dir) / 'step_100000'
      rollout_dir.mkdir()
      (rollout_dir / 'metadata.json').write_text('{}')
      self._write_trajectory(rollout_dir / 'trajectory_delay0.npz')
      np.savez_compressed(
          rollout_dir / 'trajectory_delay4.npz',
          qpos=np.zeros((2, 2, 2)),
      )
      with self.assertRaises(ValueError):
        self.render.validate_rollout_dir(rollout_dir)


if __name__ == '__main__':
  unittest.main()
