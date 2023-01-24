from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from common.mlp import MLP
from common.utils import default_init

# import distrax
# TODO : remove tfp  when distrax.Tanh is numerically stable
tfb = tfp.bijectors
tfd = tfp.distributions


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: float = 0.0
    final_fc_init_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:  # distrax.Distribution:
        outputs = MLP(
            self.hidden_dims[:-1],
            output_dim=self.hidden_dims[-1],
            output_activation=nn.relu,
            dropout_rate=self.dropout_rate,
        )(observations, training=training)
        # means
        means = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(
            outputs
        )

        if self.init_mean is not None:
            means += self.init_mean

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        # std
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # base_dist = distrax.MultivariateNormalDiag(
        #     loc=means,
        #     scale_diag=jnp.exp(log_stds) * temperature
        # )
        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        if self.tanh_squash_distribution:
            # tanh numerical instability  in distrax :((((((((
            # https://github.com/deepmind/distrax/issues/7
            # return distrax.Transformed(
            #     distribution=base_dist,
            #     bijector=distrax.Block(distrax.Tanh(), ndims=1)
            # )
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return base_dist
