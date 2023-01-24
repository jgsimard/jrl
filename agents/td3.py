import functools
from typing import Sequence, Any

import jax
import numpy as np
import optax
import rlax
from flax import linen as nn
from jax import numpy as jnp

from common.mlp import MLP
from common.types import TrainState, Params, Batch
from critics.mlp import NCriticMLP
from policies import sample


def update_critic(
    actor: TrainState,
    critic: TrainState,
    batch: Batch,
    discount: float,
    rng: Any,
    policy_noise: float,
    noise_clip: float,
):
    def critic_loss_fn(critic_params):
        # noisy actions
        next_actions_det = actor.apply_fn(actor.target_params, batch.next_observations)
        noise = jnp.clip(
            jax.random.normal(rng, next_actions_det.shape) * policy_noise,
            -noise_clip,
            noise_clip,
        )
        next_actions = jnp.clip(next_actions_det + noise, -1.0, 1.0)

        # twin targets
        next_q1, next_q2 = critic.apply_fn(
            critic.target_params, batch.next_observations, next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)
        # bellman target equation
        target_q = batch.rewards + discount * batch.masks * next_q

        # estimates
        q1, q2 = critic.apply_fn(critic_params, batch.observations, batch.actions)

        loss = rlax.l2_loss(q1, target_q).mean() + rlax.l2_loss(q2, target_q).mean()
        info = {"critic_loss": loss, "q1": q1.mean(), "q2": q2.mean()}
        return loss, info

    value_and_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    (_, info), grads = value_and_grad_fn(critic.params)
    return critic.apply_gradients(grads=grads), info


def update_actor(actor: TrainState, critic: TrainState, batch: Batch):
    def actor_loss_fn(actor_params: Params):
        actions = actor.apply_fn(actor_params, batch.observations)
        # use SAC trick of using the min here
        q1, q2 = critic.apply_fn(critic.params, batch.observations, actions)
        q = jnp.minimum(q1, q2)
        loss = -q.mean()
        info = {"actor_loss": loss}
        return loss, info

    value_and_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (_, info), grads = value_and_grad_fn(actor.params)
    return actor.apply_gradients(grads=grads), info


@functools.partial(jax.jit, static_argnames=("update_target"))
def _update(
    actor: TrainState,
    critic: TrainState,
    batch: Batch,
    tau: float,
    discount: float,
    update_target: bool,
    rng: Any,
    policy_noise: float,
    noise_clip: float,
):
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(
        actor, critic, batch, discount, key, policy_noise, noise_clip
    )

    if update_target:
        new_actor, actor_info = update_actor(actor, new_critic, batch)
        new_actor = new_actor.incremental_update_target(tau)
        new_critic = new_critic.incremental_update_target(tau)
    else:
        actor_info = {}
        new_actor = actor
        new_critic = critic
    info = {**critic_info, **actor_info}
    return rng, new_actor, new_critic, info


class TD3:
    def __init__(
        self,
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
        max_grad_norm: float = -1.0,
    ):
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

        actor_model = MLP(hidden_dims=hidden_dims, output_dim=action_dim, output_activation=nn.tanh)
        self.actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_key, obs),
            target_params=actor_model.init(actor_key, obs),
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_model = NCriticMLP(hidden_dims=hidden_dims, n_critic=2)
        self.critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, obs, actions),
            target_params=critic_model.init(critic_key, obs, actions),
            tx=optax.adam(learning_rate=critic_lr),
        )

        self.step = 0

    def update(self, batch: Batch):
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
            self.noise_clip,
        )
        self.step += 1
        return {k: v.item() for k, v in info.items()}

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = sample.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature,
            distribution="det",
        )

        actions = np.asarray(actions)
        noise = np.random.normal(size=actions.shape) * self.exploration_noise * temperature
        return np.clip(actions + noise, -1.0, 1.0)
