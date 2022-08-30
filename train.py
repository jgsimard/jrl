import os

import warnings # TODO remove this

import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agents.td3 import TD3Learner
from data.replay_buffer import ReplayBuffer
from common.env.utils import make_env
from common.evaluation import evaluate

warnings.filterwarnings("ignore")


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
    # video_train_folder = os.path.join(exp_path, "video", "train")
    # video_eval_folder = os.path.join(exp_path, "video", "eval")

    summary_writer = SummaryWriter(
        os.path.join(exp_path, "test"))

    env_name = params['env_name']
    print(env_name)
    env = make_env(env_name, params['seed'])
    eval_env = make_env(env_name, params['seed'] + 69)

    agent = TD3Learner(
        params['seed'],
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis]
    )
    replay_buffer = ReplayBuffer(env.observation_space,
                                 env.action_space,
                                 params['replay_buffer_size'])
    eval_returns =[]
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

        if i % params['eval_interval'] == 0:
            eval_stats = evaluate(agent, eval_env, params['eval_episodes'])

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
            #            eval_returns,
            #            fmt=['%d', '%.1f'])


if __name__ == "__main__":
    main()
