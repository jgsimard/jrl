import functools
from typing import Sequence, Any

import jax.random
import numpy as np
import optax

from flax import linen as nn
from flax.training import train_state
from flax.training.train_state import TrainState
from jax import numpy as jnp

from critics.mlp import NCriticMLP
from common.mlp import MLP
from common.utils import soft_update
from policies import sample

def critic_loss_fn(actor_target: TrainState,
                   critic: TrainState,
                   critic_params,
                   critic_target: TrainState,
                   batch: float,
                   discount: float,
                   rng: Any,
                   policy_noise: float,
                   noise_clip: float):
    # estimates
    q1, q2 = critic.apply_fn(critic_params, batch.observations, batch.actions)

    # targets
    # noisy actions
    next_actions = actor_target.apply_fn(actor_target.params, batch.next_observations)
    rng, key_noise = jax.random.split(rng)
    noise = jnp.clip(jax.random.normal(key_noise, next_actions.shape) * policy_noise,
                     -noise_clip,
                     noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1, 1)

    # twin targets
    next_q1, next_q2 = critic_target.apply_fn(critic_target.params,
                                              batch.next_observations,
                                              next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    # bellman target equation
    target_q = batch.rewards + discount * batch.masks * next_q

    # TODO : use rlax here
    loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
    return loss, rng


def update_critic(actor_target: TrainState,
                  critic: TrainState,
                  target_critic: TrainState,
                  data,
                  discount: float,
                  rng: Any,
                  policy_noise: float,
                  noise_clip: float):
    grad_fn = jax.grad(critic_loss_fn, argnums=2, has_aux=True)
    grads, rng = grad_fn(actor_target, critic, critic.params, target_critic, data, discount,
                         rng, policy_noise, noise_clip)
    return critic.apply_gradients(grads=grads), rng


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
            actor_target: TrainState,
            critic: TrainState,
            critic_target: TrainState,
            batch,
            tau: float,
            discount: float,
            update_target: bool,
            rng: Any,
            policy_noise: float,
            noise_clip: float):
    new_critic, rng = update_critic(actor_target, critic, critic_target, batch, discount,
                                    rng, policy_noise, noise_clip)

    if update_target:
        new_actor = update_actor(actor, new_critic, batch)

        new_critic_target = soft_update(new_critic, critic_target, tau)
        new_actor_target = soft_update(new_actor, actor_target, tau)
    else:
        new_actor = actor
        new_critic_target = critic_target
        new_actor_target = actor_target

    return new_actor, new_actor_target, new_critic, new_critic_target, rng


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
                 target_update_period: int = 2,
                 exploration_noise: float = 0.1,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5):
        action_dim = actions.shape[-1]

        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_model = MLP(
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            output_activation=nn.tanh)
        actor_params = actor_model.init(actor_key, obs)
        self.actor = train_state.TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )

        actor_target_params = actor_model.init(actor_key, obs)
        self.actor_target = train_state.TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_target_params,
            tx=optax.identity()
        )

        critc_model = NCriticMLP(hidden_dims=hidden_dims, n_critic=2)
        critic_params = critc_model.init(critic_key, obs, actions)
        self.critic = train_state.TrainState.create(
            apply_fn=critc_model.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr)
        )

        critic_target_params = critc_model.init(critic_key, obs, actions)
        self.critic_target = train_state.TrainState.create(
            apply_fn=critc_model.apply,
            params=critic_target_params,
            tx=optax.identity()
        )

        self.rng = rng
        self.step = 1

    def update(self, batch):
        update_target = self.step % self.target_update_period == 0
        self.actor, self.actor_target, self.critic, self.critic_target, self.rng = _update(
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
            batch,
            self.tau,
            self.discount,
            update_target,
            self.rng,
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
