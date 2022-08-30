import functools
from typing import Sequence, Any

import jax
import numpy as np
import optax
from flax import linen as nn
from jax import numpy as jnp

from common.mlp import MLP
from common.types import TrainState
# from common.utils import soft_target_update
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
    next_actions = jnp.clip(next_actions_det + noise, -1, 1)

    # twin targets
    next_q1, next_q2 = critic.apply_fn(critic.target_params,
                                       batch.next_observations,
                                       next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    # bellman target equation
    target_q = batch.rewards + discount * batch.masks * next_q

    # estimates
    q1, q2 = critic.apply_fn(critic_params, batch.observations, batch.actions)

    # TODO : use rlax here
    loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
    return loss


def update_critic(actor: TrainState,
                  critic: TrainState,
                  data,
                  discount: float,
                  rng: Any,
                  policy_noise: float,
                  noise_clip: float):
    grad_fn = jax.grad(critic_loss_fn, argnums=2)
    grads = grad_fn(actor, critic, critic.params, data, discount,
                         rng, policy_noise, noise_clip)
    return critic.apply_gradients(grads=grads)


def actor_loss_fn(actor: TrainState,
                  actor_params,
                  critic: TrainState,
                  batch):
    actions = actor.apply_fn(actor_params, batch.observations)
    q1, q2 = critic.apply_fn(critic.params, batch.observations, actions)
    q = jnp.minimum(q1, q2)
    loss = -q.mean()
    return loss


def update_actor(actor: TrainState, critic: TrainState, batch):
    grad_fn = jax.grad(actor_loss_fn, argnums=1)
    grads = grad_fn(actor, actor.params, critic,  batch)
    return actor.apply_gradients(grads=grads)


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
    new_critic = update_critic(actor, critic, batch, discount,
                                    rng, policy_noise, noise_clip)

    if update_target:
        new_actor = update_actor(actor, new_critic, batch)

        new_actor = new_actor.replace(
            target_params=optax.incremental_update(new_actor.params, new_actor.target_params, tau)
        )
        new_critic = new_critic.replace(
            target_params=optax.incremental_update(new_critic.params, new_critic.target_params, tau)
        )
        # new_actor = soft_target_update(new_actor, tau)
        # new_critic = soft_target_update(critic, tau)
    else:
        new_actor = actor
        new_critic = critic

    return new_actor, new_critic


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
                 noise_clip: float = 0.5):
        action_dim = actions.shape[-1]

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

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


        critc_model = NCriticMLP(hidden_dims=hidden_dims, n_critic=2)
        self.critic = TrainState.create(
            apply_fn=critc_model.apply,
            params=critc_model.init(critic_key, obs, actions),
            target_params=critc_model.init(critic_key, obs, actions),
            tx=optax.adam(learning_rate=critic_lr)
        )


        self.rng = rng
        self.step = 1

    def update(self, batch):
        update_target = self.step % self.policy_freq == 0
        self.rng, update_key = jax.random.split(self.rng)
        self.actor, self.critic = _update(
            self.actor,
            self.critic,
            batch,
            self.tau,
            self.discount,
            update_target,
            update_key,
            self.policy_noise,
            self.noise_clip
        )
        self.step += 1
        return {'info': 0}

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature,
            distribution='det')
        self.rng = rng

        actions = np.asarray(actions)
        noise = np.random.normal(size=actions.shape) * self.exploration_noise * temperature
        actions = actions + noise
        return np.clip(actions, -1, 1)
