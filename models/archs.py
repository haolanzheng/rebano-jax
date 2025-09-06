import numpy as np 
import jax 
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.module import Module, compact
from jax.nn.initializers import glorot_normal

from typing import Any, Callable

from .layers import get_dense, PirateBlock, FourierEnc    


class MLP(Module):
    """
    Multi‑layer perceptron with optional weight factorization (WeightNorm).

    Args:
        num_layers: Number of Dense layers (including the output layer).
        hidden_dim: Width of each hidden layer.
        out_dim:    Dimension of the final output layer.
        act:        Activation function
        init_fn:    Initializer for weights.
        factorized: If True, each Dense layer is wrapped with flax.linen.WeightNorm,
                    resulting in column‑wise weight factorization (g * v / ||v||).
    """
    num_layers: int = 4
    hidden_dim: int = 32
    out_dim:    int = 1
    act:        Callable = nn.tanh
    init_fn:    Any = glorot_normal()
    factorized: bool = True

    @compact
    def __call__(self, x):
        
        y = x
        
        for i in range(self.num_layers - 1):
            y = get_dense(self.hidden_dim, self.init_fn, self.factorized)(y)
            y = self.act(y)

        y = get_dense(self.out_dim, self.init_fn, self.factorized)(y)
        return y

class PirateNet(Module):
    """
    PirateNet implementation
    
    Args:
    num_blocks: Number of PirateNet blocks.
    embed_dim:  Dimension of the embedded coordinates.
    hidden_dim: Width of each hidden layer.
    out_dim:    Dimension of the final output layer.
    act:        Activation function
    init_fn:    Initializer for weights.
    factorized: If True, each Dense layer is wrapped with flax.linen.WeightNorm,
                    resulting in column‑wise weight factorization (g * v / ||v||).
    """
    num_blocks:  int = 2
    embed_scale: float = 0.1
    embed_dim:   int = 128
    hidden_dim:  int = 32
    out_dim:     int = 1
    act:         Callable = nn.tanh
    init_fn:     Any = glorot_normal()
    factorized:  bool = True
    
    @compact
    def __call__(self, x):
        y = x
        
        y = FourierEnc(embed_scale=self.embed_scale, embed_dim=self.embed_dim)(y)
        
        U = get_dense(self.hidden_dim, self.init_fn, self.factorized, name='gate_U')(y)
        V = get_dense(self.hidden_dim, self.init_fn, self.factorized, name='gate_V')(y)
        
        U, V = self.act(U), self.act(V)
        
        y = get_dense(self.hidden_dim, self.init_fn, self.factorized)(y)
        
        for i in range(self.num_blocks):
            block = PirateBlock(num_layers=3, 
                                hidden_dim=self.hidden_dim, 
                                act=self.act, 
                                init_fn=self.init_fn, 
                                factorized=self.factorized,
                                name=f'block{i+1}')
            y = block(y, U, V)
        
        y = get_dense(self.out_dim, self.init_fn, self.factorized, name='w_out', use_bias=False)(y)
        
        return y
