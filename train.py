import os
import random

import warnings  # TODO remove this

import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agents.td3 import TD3
from agents.sac import SAC
from data.replay_buffer import ReplayBuffer
from common.env.utils import make_env
from common.evaluation import evaluate

warnings.filterwarnings("ignore")  # TODO remove this


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
    agent_name = params['agent_name']

    print(agent_name)
    print(env_name)

    summary_writer = SummaryWriter(os.path.join(exp_path, agent_name, env_name))

    video_train_folder = os.path.join(exp_path, "video", "train") if params['save_video'] else None
    video_eval_folder = os.path.join(exp_path, "video", "eval") if params['save_video'] else None
    env = make_env(env_name, params['seed'], video_train_folder)
    eval_env = make_env(env_name, params['seed'] + 69, video_eval_folder)

    # seeding
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # agent
    if agent_name == 'TD3':
        agent = TD3(
            params['seed'],
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            actor_lr=params['TD3']['actor_lr'],
            critic_lr=params['TD3']['critic_lr'],
            exploration_noise=params['TD3']['exploration_noise'],
            policy_noise=params['TD3']['exploration_noise'],
            noise_clip=params['TD3']['noise_clip']
        )
    elif agent_name == 'SAC':
        agent = SAC(
            params['seed'],
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            actor_lr=params['SAC']['actor_lr'],
            critic_lr=params['SAC']['critic_lr'],
            temperature_lr=params['SAC']['temperature_lr']
        )
    else:
        raise NotImplementedError(f"Agent {agent_name} not implemented yet")
    replay_buffer = ReplayBuffer(env.observation_space,
                                 env.action_space,
                                 params['replay_buffer_size'])
    eval_returns = []
    done = False
    observation = env.reset()
    print("observation", type(observation), observation.shape)

    for i in (pbar:= tqdm.tqdm(range(1, params['max_steps'] + 1),
                       smoothing=0.1,
                       disable=not params['tqdm'])):
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


if __name__ == "__main__":
    # # force jax to be on the cpu!
    # jax.config.update('jax_platform_name', 'cpu')

    # because of my small gpu
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

    main()
