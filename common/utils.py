import jax
from flax.training.train_state import TrainState


def soft_update(state: TrainState, state_target: TrainState, tau):
    updated_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau),
                                  state.params,
                                  state_target.params)
    return state_target.replace(params=updated_params)
