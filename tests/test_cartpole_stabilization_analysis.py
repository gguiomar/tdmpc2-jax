from scripts.analyze_cartpole_stabilization_scout import plateau_summary


def test_plateau_summary_finds_sustained_and_rolling_plateaus():
  points = [
      (2500, 100.0),
      (5000, 300.0),
      (7500, 600.0),
      (10000, 850.0),
      (12500, 880.0),
      (15000, 884.0),
      (17500, 882.0),
      (20000, 883.0),
      (22500, 881.0),
      (25000, 884.0),
  ]
  summary = plateau_summary(points)
  assert summary['sustained_plateau_step'] == 10000
  assert summary['rolling_plateau_step'] == 10000
  assert summary['rolling_plateau_episode_equivalent'] == 20.0
  assert summary['rolling_plateau_episode_cycles_per_env'] == 2.5


def test_plateau_summary_can_report_no_stable_window():
  points = [(step, 100.0 + 50.0 * index) for index, step in enumerate(range(2500, 27500, 2500))]
  summary = plateau_summary(points)
  assert summary['rolling_plateau_step'] is None
