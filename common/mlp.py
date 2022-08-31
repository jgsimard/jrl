from typing import Sequence, Callable, Union

from flax import linen as nn
import jax.numpy as jnp


# kernel_init=nn.initializers.he_uniform()

def dense_layer(x, size, activation, dropout_rate=0.0, training=True, layer_norm=False):
    x = nn.Dense(size, kernel_init=nn.initializers.he_normal())(x)
    # x = nn.Dense(size)(x)

    if dropout_rate > 0:
        x = nn.Dropout(rate=dropout_rate)(x, deterministic=not training)
    if layer_norm:
        x = nn.LayerNorm()(x)
    if activation is not None:
        x = activation(x)
    return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activations: Callable = nn.relu
    output_activation: Union[Callable, None] = None
    layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        for size in self.hidden_dims:
            x = dense_layer(x, size, self.activations, self.dropout_rate, training, self.layer_norm)
        x = dense_layer(x, self.output_dim, self.output_activation,
                        self.dropout_rate, training, self.layer_norm)
        return x
