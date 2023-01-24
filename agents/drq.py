import functools
from typing import Sequence, Any, Optional, Tuple

import jax
import numpy as np
import optax

import distrax
from flax import linen as nn
from jax import numpy as jnp

from agents.sac import update_actor, update_critic, Temperature, update_temperature
from common.augmentations import batched_random_crop
from common.types import TrainState, Batch
from common.utils import default_init
from critics.mlp import NCriticMLP
from policies.normal_tanh import NormalTanhPolicy
from policies.sample import sample_actions


# networks
class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0 - 0.5
        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(3, 3),
                strides=(stride, stride),
                kernel_init=default_init(),  # why this one?
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class DrQNCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    latent_dim: int = 50
    n_critic: int = 2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features, self.cnn_strides, self.cnn_padding, name="SharedEncoder")(
            observations
        )

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NCriticMLP(self.hidden_dims, n_critic=self.n_critic)(x, actions)


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        x = Encoder(self.cnn_features, self.cnn_strides, self.cnn_padding, name="SharedEncoder")(
            observations
        )

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x, temperature)


@functools.partial(jax.jit, static_argnames=("backup_entropy", "update_target"))
def _update(
    rng: Any,
    actor: TrainState,
    critic: TrainState,
    temperature: TrainState,
    batch: Batch,
    tau: float,
    discount: float,
    update_target: bool,
    target_entropy: float,
    backup_entropy: bool,
):
    rng, key_aug, key_critic, key_actor = jax.random.split(rng, 4)

    # augmentationS
    observations = batched_random_crop(key_aug, batch.observations)
    next_observations = batched_random_crop(key_aug, batch.next_observations)
    batch.replace(observations=observations, next_observations=next_observations)

    # update critic using SAC update
    new_critic, critic_info = update_critic(
        key_critic,
        actor,
        critic,
        temperature,
        batch,
        discount,
        backup_entropy=backup_entropy,
    )

    if update_target:
        new_critic = new_critic.incremental_update_target(tau)
    else:
        new_critic = critic

    # extra depth is due to params / target_params in custom TrainState
    new_actor_params = actor.params["params"].copy(
        {"SharedEncoder": new_critic.params["params"]["SharedEncoder"]}
    )
    actor = actor.replace(params=actor.params.copy({"params": new_actor_params}))

    # update actor, temperature using SAC functions
    new_actor, actor_info = update_actor(key_actor, actor, new_critic, temperature, batch)
    new_temperature, temperature_info = update_temperature(
        temperature, actor_info["entropy"], target_entropy
    )

    info = {**critic_info, **actor_info, **temperature_info}
    return rng, new_actor, new_critic, new_temperature, info


class DrQ:
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_freq: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        # init_mean: Optional[np.ndarray] = None,
        # policy_final_fc_init_scale: float = 1.0
    ):
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_model = DrQPolicy(
            hidden_dims,
            action_dim,
            cnn_features,
            cnn_strides,
            cnn_padding,
            latent_dim,
        )
        self.actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_key, observations),
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_model = DrQNCritic(
            hidden_dims=hidden_dims,
            cnn_features=cnn_features,
            cnn_strides=cnn_strides,
            cnn_padding=cnn_padding,
            latent_dim=latent_dim,
            n_critic=2,
        )
        self.critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, observations, actions),
            target_params=critic_model.init(critic_key, observations, actions),
            tx=optax.adam(learning_rate=critic_lr),
        )

        temperature_model = Temperature(init_temperature)
        self.temperature = TrainState.create(
            apply_fn=temperature_model.apply,
            params=temperature_model.init(temp_key),
            tx=optax.adam(learning_rate=temperature_lr),
        )

        self.step = 0

    def update(self, batch):
        update_target = self.step % self.policy_freq == 0
        self.rng, self.actor, self.critic, self.temperature, info = _update(
            self.rng,
            self.actor,
            self.critic,
            self.temperature,
            batch,
            self.tau,
            self.discount,
            update_target,
            self.target_entropy,
            self.backup_entropy,
        )
        self.step += 1
        return {k: v.item() for k, v in info.items()}

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        actions = np.asarray(actions)
        return np.clip(actions, -1.0, 1.0)
