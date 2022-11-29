# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

import copy
from typing import OrderedDict

import dm_env
import numpy as np
from dm_control import suite
from gym import core, spaces

from envs.wrappers.common import TimeStep


def dmc_spec2gym_space(spec):
    if isinstance(spec, (OrderedDict, dict)):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    if isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(low=spec.minimum,
                          high=spec.maximum,
                          shape=spec.shape,
                          dtype=spec.dtype)
    if isinstance(spec, dm_env.specs.Array):
        return spaces.Box(low=-float('inf'),
                          high=float('inf'),
                          shape=spec.shape,
                          dtype=spec.dtype)
    raise NotImplementedError


class DMCEnv(core.Env):

    def __init__(self,
                 domain_name: str | None = None,
                 task_name: str | None = None,
                 env: dm_env.Environment | None = None,
                 task_kwargs: dict = {},
                 environment_kwargs=None):
        assert 'random' in task_kwargs, 'Please specify a seed, for deterministic behaviour.'
        assert (
            env is not None
            or (domain_name is not None and task_name is not None)
        ), 'You must provide either an environment or domain and task names.'

        if env is None:
            env = suite.load(domain_name=domain_name,
                             task_name=task_name,
                             task_kwargs=task_kwargs,
                             environment_kwargs=environment_kwargs)

        self._env = env
        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(
            self._env.observation_spec())

        self.seed(seed=task_kwargs['random'])

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation

        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True

        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', f'only support rgb_array mode, given {mode}'
        return self._env.physics.render(height=height,
                                        width=width,
                                        camera_id=camera_id)
