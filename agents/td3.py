import functools
from typing import Sequence, Any

import jax
import numpy as np
import optax
from flax import linen as nn
from jax import numpy as jnp
import rlax

from common.mlp import MLP
from common.types import TrainState
from critics.mlp import NCriticMLP
from policies import sample


def critic_loss_fn(actor: TrainState,
                   critic: TrainState,
                   critic_params,
                   batch: float,
                   discount: float,
                   rng: Any,
                   policy_noise: float,
                   noise_clip: float):

    # noisy actions
    next_actions_det = actor.apply_fn(actor.target_params, batch.next_observations)
    noise = jnp.clip(jax.random.normal(rng, next_actions_det.shape) * policy_noise,
                     -noise_clip,
                     noise_clip)
    next_actions = jnp.clip(next_actions_det + noise, -1.0, 1.0)

    # twin targets
    next_q1, next_q2 = critic.apply_fn(critic.target_params,
                                       batch.next_observations,
                                       next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    # bellman target equation
    target_q = batch.rewards + discount * batch.masks * next_q

    # estimates
    q1, q2 = critic.apply_fn(critic_params, batch.observations, batch.actions)

    q1_loss = rlax.l2_loss(q1, target_q).mean()
    q2_loss = rlax.l2_loss(q2, target_q).mean()
    loss = q1_loss + q2_loss
    return loss


def update_critic(actor: TrainState,
                  critic: TrainState,
                  batch,
                  discount: float,
                  rng: Any,
                  policy_noise: float,
                  noise_clip: float):
    value_and_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=2)
    critic_loss, grads = value_and_grad_fn(
        actor, critic, critic.params, batch, discount, rng, policy_noise, noise_clip)
    return critic.apply_gradients(grads=grads), critic_loss


def actor_loss_fn(actor: TrainState,
                  actor_params,
                  critic: TrainState,
                  batch):
    actions = actor.apply_fn(actor_params, batch.observations)
    q1, _ = critic.apply_fn(critic.params, batch.observations, actions)
    loss = -q1.mean()
    return loss


def update_actor(actor: TrainState, critic: TrainState, batch):
    value_and_grad_fn = jax.value_and_grad(actor_loss_fn, argnums=1)
    actor_loss, grads = value_and_grad_fn(actor, actor.params, critic,  batch)
    return actor.apply_gradients(grads=grads), actor_loss


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update(actor: TrainState,
            critic: TrainState,
            batch,
            tau: float,
            discount: float,
            update_target: bool,
            rng: Any,
            policy_noise: float,
            noise_clip: float):
    rng, key = jax.random.split(rng)
    new_critic, critic_loss = update_critic(actor, critic, batch, discount,
                                    key, policy_noise, noise_clip)

    if update_target:
        new_actor, actor_loss = update_actor(actor, new_critic, batch)

        new_actor = new_actor.replace(
            target_params=optax.incremental_update(new_actor.params, new_actor.target_params, tau)
        )
        new_critic = new_critic.replace(
            target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
        )
    else:
        actor_loss = jnp.empty_like(critic_loss)
        new_actor = actor
        new_critic = critic
    return rng, new_actor, new_critic, {"critic_loss": critic_loss, 'actor_loss': actor_loss}


class TD3Learner:
    def __init__(self,
                 seed: int,
                 obs: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 policy_freq: int = 2,
                 exploration_noise: float = 0.1,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 max_grad_norm: float = -1.0):
        action_dim = actions.shape[-1]

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_grad_norm = max_grad_norm

        rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_model = MLP(
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            output_activation=nn.tanh)
        self.actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_key, obs),
            target_params=actor_model.init(actor_key, obs),
            tx=optax.adam(learning_rate=actor_lr)
        )

        critic_model = NCriticMLP(hidden_dims=hidden_dims, n_critic=2)
        self.critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, obs, actions),
            target_params=critic_model.init(critic_key, obs, actions),
            tx=optax.adam(learning_rate=critic_lr)
        )

        self.step = 0

    def update(self, batch):
        update_target = self.step % self.policy_freq == 0
        self.rng, self.actor, self.critic, info = _update(
            self.actor,
            self.critic,
            batch,
            self.tau,
            self.discount,
            update_target,
            self.rng,
            self.policy_noise,
            self.noise_clip
        )
        self.step += 1
        return {k: v.item() for k, v in info.items()}

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = sample.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature,
            distribution='det')

        actions = np.asarray(actions)
        noise = np.random.normal(size=actions.shape) * self.exploration_noise * temperature
        return np.clip(actions + noise, -1.0, 1.0)
