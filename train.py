import os
import warnings  # TODO remove this
import random
import copy


import hydra
import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from data.replay_buffer import ReplayBuffer
from envs.utils import make_env
from common.evaluation import evaluate
from agents import make_agent


# to work with my tiny gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

warnings.filterwarnings("ignore")  # TODO remove this
warnings.filterwarnings("ignore", category=DeprecationWarning)  # TODO remove this


PLANET_ACTION_REPEAT = {
    "cartpole-swingup": 8,
    "reacher-easy": 4,
    "cheetah-run": 4,
    "finger-spin": 2,
    "ball_in_cup-catch": 4,
    "walker-walk": 2,
}


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
    exp_path = hydra_cfg["runtime"]["output_dir"]

    env_name = params["env_name"]
    agent_name = params["agent"]["name"]

    print(agent_name)
    print(env_name)

    summary_writer = SummaryWriter(os.path.join(exp_path, agent_name, env_name))

    # video folder
    video_train_folder = os.path.join(exp_path, "video", "train") if params["save_video"] else None
    video_eval_folder = os.path.join(exp_path, "video", "eval") if params["save_video"] else None

    # action repeat
    if params["action_repeat"] is not None:
        action_repeat = params["action_repeat"]
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(env_name, 2)

    # pixels
    if params["pixels"]:
        gray_scale = params["gray_scale"]
        image_size = params["image_size"]

        def make_env_(env_name, seed, video_folder, envpool=False):
            return make_env(
                env_name,
                seed,
                video_folder,
                action_repeat=action_repeat,
                image_size=image_size,
                frame_stack=3,
                from_pixels=True,
                gray_scale=gray_scale,
                use_envpool=envpool,
            )

    else:
        make_env_ = make_env

    env = make_env_(env_name, params["seed"], video_train_folder)
    eval_env = make_env_(env_name, params["seed"] + 69, video_eval_folder)

    # seeding
    np.random.seed(params["seed"])
    random.seed(params["seed"])

    # agent
    agent = make_agent(params, env)

    replay_buffer_type = params["replay_buffer_type"]
    if replay_buffer_type == "basic":
        replay_buffer_size = params["replay_buffer_size"] or params["max_steps"] // action_repeat
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space, replay_buffer_size)
    else:
        raise NotImplementedError(f"Replay buffer type {replay_buffer_type} not implemented yet")
    eval_returns = []
    done = False
    observation = env.reset()
    num_steps = params["max_steps"] // action_repeat + 1

    for i in (pbar := tqdm.tqdm(range(1, num_steps), smoothing=0.1, disable=not params["tqdm"])):
        if i < params["start_training"]:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)

        next_obs, reward, done, info = env.step(action)

        if env_name == "MountainCarContinuous-v0":
            reward = reward + 13 * np.abs(next_obs[1])

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done), next_obs)
        observation = next_obs

        if done:
            observation, done = env.reset(), False
            pbar.set_description(f"Episode return={info['episode']['return']:.2f}")
            for k, v in info["episode"].items():
                summary_writer.add_scalar(f"training/{k}", v, info["total"]["timesteps"])

            if "is_success" in info:
                summary_writer.add_scalar(
                    "training/success", info["is_success"], info["total"]["timesteps"]
                )

        if i >= params["start_training"]:
            for _ in range(params["updates_per_step"]):
                batch = replay_buffer.sample(params["batch_size"])
                update_info = agent.update(batch)

            if i % params["log_interval"] == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f"training/{k}", v, i)
                summary_writer.flush()

        if i % params["eval_interval"] == 0:
            eval_stats = evaluate(agent, eval_env, params["eval_episodes"])

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, info["total"]["timesteps"])
            summary_writer.flush()

            eval_returns.append((info["total"]["timesteps"], eval_stats["return"]))

        if params["reset"] and i % params["reset_interval"] == 0:
            # create a completely new agent
            new_params = copy.deepcopy(params)
            new_params["seed"] = new_params["seed"] + i
            agent = make_agent(new_params, env)


if __name__ == "__main__":
    # # force jax to be on the cpu!
    # import jax
    # jax.config.update('jax_platform_name', 'cpu')

    # because of my small gpu
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ['MUJOCO_GL'] = 'egl'
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
    # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "cpu"

    main()
