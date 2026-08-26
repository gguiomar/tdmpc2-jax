import pytest

from tdmpc2_jax.envs import mjx_dmc


@pytest.mark.parametrize('value', [0.0, -0.1, 1.01])
def test_actuator_strength_scale_rejects_out_of_range_values(monkeypatch, value):
  monkeypatch.setattr(mjx_dmc, 'mujoco', object())
  monkeypatch.setattr(mjx_dmc, 'mjx', object())
  monkeypatch.setattr(mjx_dmc, 'suite', object())
  with pytest.raises(ValueError, match='actuator_strength_scale'):
    mjx_dmc.MJXDMCBatchEnv(
        num_envs=1,
        seed=0,
        task='pendulum-swingup',
        actuator_strength_scale=value,
    )


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('joint_damping_scale', 0.0),
        ('joint_damping_scale', float('nan')),
        ('gravity_scale', -1.0),
        ('gravity_scale', float('inf')),
        ('fixed_observation_noise_scale', -0.01),
    ],
)
def test_frontier_scale_validation_happens_before_model_load(
    monkeypatch, name, value
):
  monkeypatch.setattr(mjx_dmc, 'mujoco', object())
  monkeypatch.setattr(mjx_dmc, 'mjx', object())
  monkeypatch.setattr(mjx_dmc, 'suite', object())
  with pytest.raises(ValueError, match=name):
    mjx_dmc.MJXDMCBatchEnv(
        num_envs=1,
        seed=0,
        task='pendulum-swingup',
        **{name: value},
    )
