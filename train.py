import os

import gym
import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agents.td3 import TD3Learner
from data.replay_buffer import ReplayBuffer


# state = TrainState.create(
#           apply_fn=model.apply,
#           params=variables['params'],
#           tx=tx)
#       grad_fn = jax.grad(make_loss_fn(state.apply_fn))
#       for batch in data:
#         grads = grad_fn(state.params, batch)
#         state = state.apply_gradients(grads=grads)


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
