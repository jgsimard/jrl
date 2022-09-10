# jrl : JAX for Reinforcement Learning

## Goal
Implement as many RL algorithms as possible in JAX

## Completed 

| Algorithm | Action Space | Method | File |
|-----------|--------------|--------|------|
|[DDPG](https://arxiv.org/pdf/1509.02971.pdf)   | Continuous   |Model-Free| [ddpg.py](agents/ddpg.py)|
|[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html)   |Continuous| Model-Free  | [td3.py](agents/td3.py) |
|[SAC](https://arxiv.org/pdf/1801.01290)   | Continuous   |Model-Free |[sac.py](agents/sac.py)|
|[DrQ](https://arxiv.org/pdf/2004.13649)   | Continuous   |Model-Free |[drq.py](agents/drq.py)|
|[DroQ](http://arxiv.org/abs/2110.02034)   | Continuous   |Model-Free |[sac.py](agents/droq.py)|

## Whishlist

| Algorithm | Action Space | Method | File |
|-----------|--------------|--------|------|
|[DQN](https://arxiv.org/pdf/1312.5602)   | Discrete   |Model-Free| |
|[Rainbow](https://arxiv.org/pdf/1710.02298)   |Discrete| Model-Free  | |
|[Planet](https://arxiv.org/pdf/1811.04551.pdf)   | Continuous/Discrete   |Model-based ||
|[Dreamer](https://arxiv.org/pdf/1912.01603)   | Continuous/Discrete   |Model-based ||
|[DreamerV2](https://arxiv.org/pdf/2010.02193)   | Continuous/Discrete   |Model-based ||
|[TRPO](https://arxiv.org/pdf/1502.05477)   | Continuous/Discrete   |Model-based ||
|[PPO](http://arxiv.org/abs/1707.06347)   | Continuous/Discrete   |Model-based ||
|[DrQv2](https://arxiv.org/pdf/2107.09645.pdf)   | Continuous   |Model-free ||


## Sources
- Jax tutorials (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html, https://jax.readthedocs.io/en/latest/jax-101/index.html)
- JaxRL : repo with Jax implementation of a RL algorithms (https://github.com/ikostrikov/jaxrl/tree/main/jaxrl)
