import functools
from typing import Sequence, Any, Optional

import jax
import numpy as np
import optax
import rlax
from flax import linen as nn
from jax import numpy as jnp

from common.types import TrainState, Batch
from critics.mlp import NCriticMLP
from policies.normal_tanh import NormalTanhPolicy
from policies.sample import sample_actions


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update_temperature(temperature: TrainState, entropy: float, target_entropy: float):
    def temperature_loss_fn(params):
        temperature_value = temperature.apply_fn(params)
        temperature_loss = temperature_value * (entropy - target_entropy).mean()
        return temperature_loss, {'temperature': temperature_value, 'temp_loss': temperature_loss}

    value_and_grad_fn = jax.value_and_grad(temperature_loss_fn, has_aux=True)
    (_, info), grads = value_and_grad_fn(temperature.params)
    return temperature.apply_gradients(grads=grads), info


def critic_loss_fn(key: Any,
                   actor: TrainState,
                   critic: TrainState,
                   critic_params,
                   temperature: TrainState,
                   batch: Batch,
                   discount: float,
                   backup_entropy: bool):
    # noisy actions
    dist = actor.apply_fn(actor.params, batch.next_observations)
    next_actions = dist.sample(seed=key)

    # twin targets
    next_q1, next_q2 = critic.apply_fn(critic.target_params,
                                       batch.next_observations,
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
    q1, q2 = critic.apply_fn(critic_params, batch.observations, batch.actions)

    q1_loss = rlax.l2_loss(q1, target_q).mean()
    q2_loss = rlax.l2_loss(q2, target_q).mean()
    loss = q1_loss + q2_loss

    info = {'critic_loss': loss, 'q1': q1.mean(), 'q2': q2.mean()}

    return loss, info


def update_critic(key: Any,
                  actor: TrainState,
                  critic: TrainState,
                  temperature: TrainState,
                  batch: Batch,
                  discount: float,
                  backup_entropy: bool):
    value_and_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=3, has_aux=True)
    (_, info), grads = value_and_grad_fn(
        key, actor, critic, critic.params, temperature, batch, discount, backup_entropy)
    return critic.apply_gradients(grads=grads), info


def actor_loss_fn(key,
                  actor: TrainState,
                  actor_params,
                  critic: TrainState,
                  temperature: TrainState,
                  batch: Batch):
    dist_actions = actor.apply_fn(actor_params, batch.observations)
    actions = dist_actions.sample(seed=key)
    q1, q2 = critic.apply_fn(critic.params, batch.observations, actions)
    q = jnp.minimum(q1, q2)
    log_probs = dist_actions.log_prob(actions)
    loss = (log_probs * temperature.apply_fn(temperature.params) - q).mean()
    info = {'actor_loss': loss, 'entropy': -log_probs.mean()}
    return loss, info


def update_actor(key, actor: TrainState, critic: TrainState, temperature: TrainState, batch):
    value_and_grad_fn = jax.value_and_grad(actor_loss_fn, argnums=2, has_aux=True)
    (_, info), grads = value_and_grad_fn(key, actor, actor.params, critic, temperature, batch)
    return actor.apply_gradients(grads=grads), info


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'update_target'))
def _update(rng: Any,
            actor: TrainState,
            critic: TrainState,
            temperature: TrainState,
            batch: Batch,
            tau: float,
            discount: float,
            update_target: bool,
            target_entropy: float,
            backup_entropy: bool):
    rng, key_critic, key_actor = jax.random.split(rng, 3)
    new_critic, critic_info = update_critic(key_critic, actor, critic, temperature, batch, discount,
                                            backup_entropy=backup_entropy)

    if update_target:
        new_critic = new_critic.replace(
            target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
        )
    else:
        new_critic = new_critic

    new_actor, actor_info = update_actor(key_actor, actor, new_critic, temperature, batch)
    new_temperature, temperature_info = update_temperature(temperature,
                                                           actor_info['entropy'],
                                                           target_entropy)

    info = {**critic_info, **actor_info, **temperature_info}
    return rng, new_actor, new_critic, new_temperature, info


class SAC:
    def __init__(self,
                 seed: int,
                 obs: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temperature_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 policy_freq: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0
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

        actor_model = NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale
        )
        self.actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_key, obs),
            tx=optax.adam(learning_rate=actor_lr)
        )

        critic_model = NCriticMLP(hidden_dims=hidden_dims, n_critic=2)
        self.critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, obs, actions),
            target_params=critic_model.init(critic_key, obs, actions),
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
            self.backup_entropy
        )
        self.step += 1
        return {k: v.item() for k, v in info.items()}

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature)
        actions = np.asarray(actions)
        return np.clip(actions, -1.0, 1.0)
