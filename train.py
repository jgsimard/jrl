import os
import random
import copy

import warnings  # TODO remove this

import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
# from dm_env import specs

from agents.td3 import TD3
from agents.sac import SAC
from agents.drq import DrQ
from data.replay_buffer import ReplayBuffer
# from data.replay_buffer_compressed import ReplayBufferStorage, make_replay_loader
from common.env.utils import make_env
from common.evaluation import evaluate

warnings.filterwarnings("ignore")  # TODO remove this


PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Entry point
    """
    print(OmegaConf.to_yaml(cfg))
    params = vars(cfg)
    for key, value in cfg.items():
        params[key] = value
    # print(params)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    exp_path = hydra_cfg['runtime']['output_dir']

    env_name = params['env_name']
    agent_name = params['agent']['name']

    print(agent_name)
    print(env_name)

    summary_writer = SummaryWriter(os.path.join(exp_path, agent_name, env_name))

    # video folder
    video_train_folder = os.path.join(exp_path, "video", "train") if params['save_video'] else None
    video_eval_folder = os.path.join(exp_path, "video", "eval") if params['save_video'] else None

    # action repeat
    if params['action_repeat'] is not None:
        action_repeat = params['action_repeat']
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(env_name, 2)

    # pixels
    if params['pixels']:
        gray_scale = params['gray_scale']
        image_size = params['image_size']


        def make_env_(env_name, seed, video_folder, envpool):
            return make_env(env_name,
                            seed,
                            video_folder,
                            action_repeat=action_repeat,
                            image_size=image_size,
                            frame_stack=3,
                            from_pixels=True,
                            gray_scale=gray_scale,
                            use_envpool=envpool)
    else:
        make_env_ = make_env

    env = make_env_(env_name, params['seed'], video_train_folder)
    eval_env = make_env_(env_name, params['seed'] + 69, video_eval_folder)

    # seeding
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # agent
    agent = make_agent(params, env)

    replay_buffer_type = params['replay_buffer_type']
    if replay_buffer_type =='basic':
        replay_buffer_size = params['replay_buffer_size'] or params['max_steps'] // action_repeat
        replay_buffer = ReplayBuffer(env.observation_space,
                                     env.action_space,
                                     replay_buffer_size)
    # elif replay_buffer_type == 'compressed':
    #     data_specs = (env.observation_spec(),
    #                   env.action_spec(),
    #                   specs.Array((1,), np.float32, 'reward'),
    #                   specs.Array((1,), np.float32, 'discount'))
    #
    #     buffer_path = os.path.join(exp_path, 'buffer')
    #
    #     replay_storage = ReplayBufferStorage(data_specs, buffer_path)
    #     replay_loader = make_replay_loader(
    #         buffer_path, params['replay_buffer_size'],
    #         params['batch_size'], params['replay_buffer_num_workers'],
    #         params['save_snapshot'], params['nstep'], params['discount'])
    #     _replay_iter = None
    else:
        raise NotImplementedError(f"Replay buffer type {replay_buffer_type} not implemented yet")
    eval_returns = []
    done = False
    observation = env.reset()
    num_steps = params['max_steps'] // action_repeat + 1

    for i in (pbar := tqdm.tqdm(range(1, num_steps), smoothing=0.1, disable=not params['tqdm'])):
        if i < params['start_training']:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)

        next_obs, reward, done, info = env.step(action)

        if env_name == "MountainCarContinuous-v0":
            reward = reward + 13 * np.abs(next_obs[1])

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done), next_obs)
        observation = next_obs

        if done:
            observation, done = env.reset(), False
            pbar.set_description(f"Episode return={info['episode']['return']:.2f}")
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            if 'is_success' in info:
                summary_writer.add_scalar('training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])

        if i >= params['start_training']:
            for _ in range(params['updates_per_step']):
                batch = replay_buffer.sample(params['batch_size'])
                update_info = agent.update(batch)

            if i % params['log_interval'] == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % params['eval_interval'] == 0:
            eval_stats = evaluate(agent, eval_env, params['eval_episodes'])

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))

        if params['reset'] and i % params['reset_interval'] == 0:
            # create a completely new agent
            new_params = copy.deepcopy(params)
            new_params['seed'] = new_params['seed'] + i
            agent = make_agent(new_params, env)


if __name__ == "__main__":
    # # force jax to be on the cpu!
    # jax.config.update('jax_platform_name', 'cpu')

    # because of my small gpu
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

    main()
