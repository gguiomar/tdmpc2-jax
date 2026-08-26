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
