import numpy as np 
import jax 
import jax.numpy as jnp

import flax.linen as nn
from flax.linen import Dense, WeightNorm
from flax.linen.initializers import zeros
from flax.linen.module import Module, compact
from jax.nn.initializers import normal, glorot_normal

from typing import Any, Callable

def get_dense(
              features:   int, 
              init_fn:    Any = glorot_normal(), 
              factorized: bool = True, 
              name:       str = None, 
              use_bias:   bool = True
              ):
            base = Dense(features, kernel_init=init_fn, use_bias=use_bias, name=name)
            if factorized:
                return WeightNorm(base, variable_filter={'kernel'})
            return base

class PirateBlock(Module):
    """
    One residual block in PirateNet
    
    Args:
        num_layers: Number of layers inside each block
        hidden_dim: Width of each layer.
        act:        Activation function.
        init_fn:    Kernel initializer.
        factorized: If True, each Dense layer is wrapped with flax.linen.WeightNorm,
                    resulting in column‑wise weight factorization (g * v / ||v||).
    """
    num_layers: int = 3
    hidden_dim: int = 32
    act:        Callable = nn.tanh
    init_fn:    Any = glorot_normal()
    factorized: bool = True
    
    @compact 
    def __call__(self, x, U, V):
        
        alpha = self.param('alpha', zeros, ())
        
        y = x
        
        for i in range(self.num_layers - 1):
            y = get_dense(self.hidden_dim, self.init_fn, self.factorized)(y)
            y = self.act(y)
            y = y * U + (1.0 - y) * V 

        y = get_dense(self.hidden_dim, self.init_fn, self.factorized)(y)
        y = self.act(y)
        y = alpha * y + (1.0 - alpha) * x
        
        return y
    
class FourierEnc(Module):
    """
    Gaussian random Fourier features
    
    Args:
        embed_scale: the standard deviation of the normal distribution from which the encoding matrix B is sampled.
        embed_dim  : embedding dimension.
    """
    embed_scale: float = 0.1
    embed_dim:   int = 128
    
    @compact
    def __call__(self, x):
        B = self.param(
            'kernel',
            lambda key, shape: normal(stddev=self.embed_scale)(key, shape),
            (x.shape[-1], self.embed_dim // 2)
        )
        x = jnp.concatenate([
            jnp.cos(x @ B),
            jnp.sin(x @ B)],
            axis=-1
        )
        
        return x
        
