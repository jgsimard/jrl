import functools
from typing import Sequence, Any, Optional, Tuple

import jax
import numpy as np
import optax
import rlax
import distrax
from flax import linen as nn
from jax import numpy as jnp
from jax.random import KeyArray

# from agents.sac import update_actor, update_critic, Temperature, update_temperature
from agents.sac import Temperature, update_temperature

from common.augmentations import batched_random_crop
from common.types import TrainState, Batch
from common.utils import default_init
from critics.mlp import NCriticMLP
from policies.normal_tanh import NormalTanhPolicy

# this version is ~5-10% slower than the other version

# networks
class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0 - 0.5
        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),  # why this one?
                        padding=self.padding)(x)
            x = nn.relu(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class DrQNCritic(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int = 50
    n_critic: int = 2

    @nn.compact
    def __call__(self,
                 representations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(self.latent_dim)(representations)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NCriticMLP(self.hidden_dims, n_critic=self.n_critic)(x, actions)


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    latent_dim: int = 50

    @nn.compact
    def __call__(self,
                 representations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        x = nn.Dense(self.latent_dim)(representations)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x, temperature)


def update_critic(key: Any, encoder: TrainState, actor: TrainState, critic: TrainState,
                  temperature: TrainState, batch: Batch, discount: float, backup_entropy: bool):
    def critic_loss_fn(critic_params, encoder_params):
        # augmentations
        representations = encoder.apply_fn(encoder_params, batch.observations)
        next_representations = encoder.apply_fn(encoder_params, batch.next_observations)

        # noisy actions
        dist = actor.apply_fn(actor.params, next_representations)
        next_actions = dist.sample(seed=key)

        # twin targets
        next_q1, next_q2 = critic.apply_fn(critic.target_params,
                                           next_representations,
                                           next_actions)
        next_q = jnp.minimum(next_q1, next_q2)

        # bellman target equation
        target_q = batch.rewards + discount * batch.masks * next_q

        # entropy
        if backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= discount * \
                        batch.masks * \
                        temperature.apply_fn(temperature.params) * \
                        next_log_probs

        # estimates
        q1, q2 = critic.apply_fn(critic_params, representations, batch.actions)

        loss = rlax.l2_loss(q1, target_q).mean() + rlax.l2_loss(q2, target_q).mean()
        info = {'critic_loss': loss, 'q1': q1.mean(), 'q2': q2.mean()}
        return loss, info

    value_and_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)
    (_, info), (grads_critic, grads_encoder) = value_and_grad_fn(critic.params, encoder.params)
    return critic.apply_gradients(grads=grads_critic), \
           encoder.apply_gradients(grads=grads_encoder), \
           info


def update_actor(key,
                 encoder: TrainState,
                 actor: TrainState,
                 critic: TrainState,
                 temperature: TrainState, batch):
    def actor_loss_fn(actor_params):
        # augmentations
        representations = encoder.apply_fn(encoder.params, batch.observations)

        # SAC
        dist_actions = actor.apply_fn(actor_params, representations)
        actions = dist_actions.sample(seed=key)
        q1, q2 = critic.apply_fn(critic.params, representations, actions)
        q = jnp.minimum(q1, q2)
        log_probs = dist_actions.log_prob(actions)
        loss = (log_probs * temperature.apply_fn(temperature.params) - q).mean()
        info = {'actor_loss': loss, 'entropy': -log_probs.mean()}
        return loss, info
    value_and_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (_, info), grads = value_and_grad_fn(actor.params)
    return actor.apply_gradients(grads=grads), info


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'update_target'))
def _update(rng: Any,
            encoder: TrainState,
            actor: TrainState,
            critic: TrainState,
            temperature: TrainState,
            batch: Batch,
            tau: float,
            discount: float,
            update_target: bool,
            target_entropy: float,
            backup_entropy: bool):
    rng, key_aug, key_critic, key_actor = jax.random.split(rng, 4)

    # augmentations
    observations = batched_random_crop(key_aug, batch.observations)
    next_observations = batched_random_crop(key_aug, batch.next_observations)
    batch.replace(observations=observations, next_observations=next_observations)

    # update critic using SAC update
    new_critic, new_encoder, critic_info = update_critic(key_critic, encoder, actor, critic,
                                                         temperature, batch, discount,
                                                         backup_entropy=backup_entropy)

    if update_target:
        new_critic = new_critic.replace(
            target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
        )
    else:
        new_critic = critic

    # update actor, temperature using SAC functions
    new_actor, actor_info = update_actor(key_actor, encoder, actor, new_critic, temperature, batch)
    new_temperature, temperature_info = update_temperature(temperature,
                                                           actor_info['entropy'],
                                                           target_entropy)

    info = {**critic_info, **actor_info, **temperature_info}
    return rng, new_encoder, new_actor, new_critic, new_temperature, info


class DrQ:
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temperature_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
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

        rng : KeyArray = jax.random.PRNGKey(seed)
        self.rng, encoder_key, actor_key, critic_key, temp_key = jax.random.split(rng, 5)

        encoder_model = Encoder(
            cnn_features,
            cnn_strides,
            cnn_padding,
        )
        self.encoder = TrainState.create(
            apply_fn=encoder_model.apply,
            params=encoder_model.init(encoder_key, observations),
            tx=optax.adam(actor_lr)
        )
        representations = self.encoder.apply_fn(self.encoder.params, observations)

        actor_model = DrQPolicy(
            hidden_dims,
            action_dim,
            latent_dim,
        )
        self.actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_key, representations),
            tx=optax.adam(learning_rate=actor_lr)
        )

        critic_model = DrQNCritic(
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            n_critic=2)
        self.critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, representations, actions),
            target_params=critic_model.init(critic_key, representations, actions),
            tx=optax.adam(learning_rate=critic_lr)
        )

        temperature_model = Temperature(init_temperature)
        self.temperature = TrainState.create(
            apply_fn=temperature_model.apply,
            params=temperature_model.init(temp_key),
            tx=optax.adam(learning_rate=temperature_lr)
        )

        self.step = 0

    def update(self, batch):
        update_target = self.step % self.policy_freq == 0
        self.rng, self.encoder, self.actor, self.critic, self.temperature, info = _update(
            self.rng,
            self.encoder,
            self.actor,
            self.critic,
            self.temperature,
            batch,
            self.tau,
            self.discount,
            update_target,
            self.target_entropy,
            self.backup_entropy
        )
        self.step += 1
        return {k: v.item() for k, v in info.items()}

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = _sample_actions(self.rng,
                                            self.encoder,
                                            self.actor,
                                            observations,
                                            temperature)
        actions = np.asarray(actions)
        return np.clip(actions, -1.0, 1.0)


@jax.jit
def _sample_actions(
        rng,
        encoder: TrainState,
        actor: TrainState,
        observations: np.ndarray,
        temperature: float = 1.0):
    representations = encoder.apply_fn(encoder.params, observations)
    dist = actor.apply_fn(actor.params, representations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
