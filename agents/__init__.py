import numpy as np

from .drq import DrQ
from .sac import SAC
from .td3 import TD3


def make_agent(params, env):
    agent_params = params['agent']
    agent_name = agent_params['name']
    if agent_name == 'TD3':
        agent = TD3(
            params['seed'],
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            actor_lr=agent_params['actor_lr'],
            critic_lr=agent_params['critic_lr'],
            exploration_noise=agent_params['exploration_noise'],
            policy_noise=agent_params['exploration_noise'],
            noise_clip=agent_params['noise_clip']
        )
    elif agent_name == 'SAC':
        agent = SAC(
            params['seed'],
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            actor_lr=agent_params['actor_lr'],
            critic_lr=agent_params['critic_lr'],
            temperature_lr=agent_params['temperature_lr']
        )
    elif agent_name == 'DroQ':
        agent = SAC(
            params['seed'],
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            actor_lr=params['DroQ']['actor_lr'],
            critic_lr=params['DroQ']['critic_lr'],
            temperature_lr=params['DroQ']['temperature_lr'],
            layer_norm=params['DroQ']['layer_norm'],
            dropout_rate=params['DroQ']['dropout_rate'],
        )
    elif agent_name == 'DrQ':
        agent = DrQ(
            params['seed'],
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            hidden_dims=agent_params['hidden_dims'],
            cnn_features=agent_params['cnn_features'],
            cnn_strides=agent_params['cnn_strides'],
            cnn_padding=agent_params['cnn_padding'],
            latent_dim=agent_params['latent_dim'],
            actor_lr=agent_params['actor_lr'],
            critic_lr=agent_params['critic_lr'],
            temperature_lr=agent_params['temperature_lr']
        )
    else:
        raise NotImplementedError(f"Agent {agent_name} not implemented yet")
    return agent
