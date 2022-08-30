from typing import Sequence

from flax import linen as nn
from jax import numpy as jnp

from common.mlp import MLP


class CriticMLP(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, obs, actions):
        obs_actions = jnp.concatenate([obs, actions])
        q = MLP(hidden_dims=self.hidden_dims,
                output_dim=1,
                layer_norm=self.layer_norm,
                dropout_rate=self.dropout_rate
                )(obs_actions)
        return q


class NCriticMLP(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    dropout_rate: float = 0.0
    n_critic: int = 2

    @nn.compact
    def __call__(self, obs, actions):
        n_critic_mlp = nn.vmap(
            CriticMLP,
            in_axes=0, out_axes=0,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            axis_size=self.n_critic)
        n_q = n_critic_mlp(self.hidden_dims, self.layer_norm, self.dropout_rate)(obs, actions)
        return n_q
