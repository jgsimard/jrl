import functools
import os
from typing import Sequence, Any

import flax.linen as nn
import gym
import hydra
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tqdm
from flax.training import train_state
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from data.replay_buffer import ReplayBuffer


def soft_update(state: TrainState, state_target: TrainState, tau):
    updated_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau),
                                  state.params,
                                  state_target.params)
    return state_target.replace(params=updated_params)
def dense_layer(x, size, activation, dropout_rate=0.0, training=True, layer_norm=False):
    x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform())(x)
    if dropout_rate > 0:
        x = nn.Dropout(rate=dropout_rate)(x, deterministic=not training)
    if layer_norm:
        x = nn.LayerNorm()(x)
    x = activation(x)
    return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activations: Any = nn.silu  # TODO fix that
    output_activation: Any = jnp.identity
    layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        for size in self.hidden_dims:
            x = dense_layer(x, size, self.activations, self.dropout_rate, training, self.layer_norm)
        x = dense_layer(x, self.output_dim, self.output_activation,
                        self.dropout_rate, training, self.layer_norm)
        return x


class CriticMLP(nn.Module):
    hidden_dims: Sequence[int]
    norm = jnp.identity
    dropout_rate = 0.0

    @nn.compact
    def __call__(self, obs, actions):
        obs_actions = jnp.concatenate([obs, actions])
        q = MLP(hidden_dims=self.hidden_dims,
                output_dim=1,
                norm=self.norm,
                dropout_rate=self.dropout_rate
                )(obs_actions)
        return q


class NCriticMLP(nn.Module):
    hidden_dims: Sequence[int]
    norm = jnp.identity
    dropout_rate = 0.0
    n_critic = 2

    @nn.compact
    def __call__(self, obs, actions):
        n_critic_mlp = nn.vmap(
            CriticMLP,
            in_axes=0, out_axes=0,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            axis_size=self.n_critic)
        n_q = n_critic_mlp(self.hidden_dims, self.norm, self.dropout_rate)(obs, actions)
        return n_q


# state = TrainState.create(
#           apply_fn=model.apply,
#           params=variables['params'],
#           tx=tx)
#       grad_fn = jax.grad(make_loss_fn(state.apply_fn))
#       for batch in data:
#         grads = grad_fn(state.params, batch)
#         state = state.apply_gradients(grads=grads)

def critic_loss_fn(actor: TrainState,
                   critic: TrainState,
                   critic_target: TrainState,
                   batch: float,
                   discount: float):
    # current estimates
    q1, q2 = critic.apply_fn(batch.obs, batch.action)

    # next estimates
    next_action = actor.apply_fn(batch.next_obs)
    next_q1, next_q2 = critic_target.apply_fn(batch.next_obs, next_action)
    next_q = jnp.minimum(next_q1, next_q2)
    target_q = batch.reward + discount * batch.mask * next_q
    # TODO use rlax here
    loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
    return loss


def update_critic(actor: TrainState,
                  critic: TrainState,
                  target_critic: TrainState,
                  data,
                  discount: float):
    return actor + critic + target_critic + data + discount

def actor_loss_fn(actor: TrainState, critic: TrainState, params, batch):
    actions = actor.apply_fn(params, batch.obs)
    q1, q2 = critic.apply_fn(batch.obs, actions)
    q = jnp.minimum(q1, q2)
    loss = -q.mean()
    return loss


def update_actor(actor: TrainState, new_critic: TrainState, batch):
    return actor + new_critic + batch


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update(actor: TrainState,
            critic: TrainState,
            critic_target: TrainState,
            batch,
            tau: float,
            discount: float,
            update_target: bool):
    new_critic = update_critic(actor, critic, critic_target, batch, discount)

    if update_target:
        new_critic_target = soft_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target
    new_actor = update_actor(actor, new_critic, batch)
    return new_actor, new_critic, new_critic_target
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
                 target_update_period: int = 1,
                 exploration_noise: float = 0.1):
        action_dim = actions.shape[-1]

        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.exploration_noise = exploration_noise

        rng = jax.random.PRNGKey(int(seed))
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
        critic_params = critc_model.init(critic_key, [obs, actions])
        self.critic = train_state.TrainState.create(
            apply_fn=critc_model.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr)
        )

        critic_target_params = critc_model.init(critic_key, [obs, actions])
        self.critic_target = train_state.TrainState.create(
            apply_fn=critc_model.apply,
            params=critic_target_params,
            tx=optax.identity()
        )

        self.rng = rng
        self.step = 1

    def update(self, batch):
        update_target = self.step % self.target_update_period == 0
        self.actor, self.critic, self.critic_target = _update(
            self.actor, self.critic, self.critic_target, batch, self.tau, update_target
        )
        self.step += 1
        return {'info': 0}

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        # rng, actions = policies.sample_actions(self.rng,
        #                                        self.actor.apply_fn,
        #                                        self.actor.params,
        #                                        observations,
        #                                        temperature,
        #                                        distribution='det')
        # self.rng = rng
        #
        # actions = np.asarray(actions)
        actions = np.ones((2,3)) + observations
        actions = actions + np.random.normal(
            size=actions.shape) * self.exploration_noise * temperature
        return np.clip(actions, -1, 1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Entry point
    """
    print(OmegaConf.to_yaml(cfg))
    params = vars(cfg)
    for key, value in cfg.items():
        params[key] = value
    print(params)

    summary_writer = SummaryWriter(
        os.path.join(os.getcwd(), "test"))

    # env = envpool.make_gym(params['env_name'], num_envs=1)
    print("XXXXXXXXx")
    print(params['env_name'])
    print()
    # doesnt work for some reason
    env = gym.make(params['env_name'])
    # env = gym.make('Pendulum-v1')


    agent = TD3Learner(
        params['seed'],
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis]
    )
    replay_buffer = ReplayBuffer(env.observation_space,
                                 env.action_space,
                                 params['replay_buffer_size'])

    done = False
    observation = env.reset()
    print(observation, type(observation), observation.shape)

    for i in tqdm.tqdm(range(1, params['max_steps'] + 1),
                       smoothing=0.1,
                       disable=not params['tqdm']):
        if i < params['start_training']:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_obs, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_obs)
        observation = next_obs

        if i >= params['start_training']:
            for _ in range(params['updates_per_step']):
                batch = replay_buffer.sample(params['batch_size'])
                update_info = agent.update(batch)

            if i % params['log_interval'] == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()


if __name__ == "__main__":
    main()
