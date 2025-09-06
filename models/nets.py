import numpy as np 
import jax 
import jax.numpy as jnp
from jax import vmap

import flax.linen as nn
from flax.linen.module import Module, compact

from ml_collections import ConfigDict
from typing import Callable, List, Any
import os, sys

from .archs import MLP, PirateNet


Array = jax.Array

class PINN(Module):
    """
    A PINN uses MLPs or PirateNets as its backbone.

    Args:
        config: PINN configurations.
    """
    config: ConfigDict
    
    @staticmethod
    def get_activation(activation: str) -> Callable:
        """Get activation function from string."""
        activation_map = {
            # Standard Flax activations
            'tanh': nn.tanh,
            'relu': nn.relu,
            'sigmoid': nn.sigmoid,
            'softplus': nn.softplus,
            'elu': nn.elu,
            'gelu': nn.gelu,
            'swish': nn.swish,
            'silu': nn.silu,
            'leaky_relu': nn.leaky_relu,
            'hard_tanh': nn.hard_tanh,
            'hard_sigmoid': nn.hard_sigmoid,
            'selu': nn.selu,
            'celu': nn.celu,
            'log_sigmoid': nn.log_sigmoid,
            'sin': jnp.sin,
            'cos': jnp.cos
        }
        
        if activation not in activation_map:
            raise NotImplementedError(f"Unknown activation: {activation}. "
                           f"Available: {list(activation_map.keys())}")
        return activation_map[activation]
    
    @staticmethod
    def get_initializer(initializer: str) -> Callable:
        """Get initializer function from string."""
        initializer_map = {
            'glorot_normal': nn.initializers.glorot_normal(),
            'glorot_uniform': nn.initializers.glorot_uniform(),
            'normal': nn.initializers.normal(),
            'he_normal': nn.initializers.he_normal(),
            'he_uniform': nn.initializers.he_uniform(),
            'xavier_normal': nn.initializers.xavier_normal(),
            'xavier_uniform': nn.initializers.xavier_uniform(),
            'zeros': nn.initializers.zeros,
            'ones': nn.initializers.ones,
        }
        
        if initializer not in initializer_map:
            raise NotImplementedError(f"Unknown initializer: {initializer}. "
                           f"Available: {list(initializer_map.keys())}")
        return initializer_map[initializer]
    
    @classmethod
    def list_activations(cls) -> list:
        """List all available activation functions."""
        return ['tanh', 'relu', 'sigmoid', 'softplus', 'elu', 'gelu',
                'swish', 'silu', 'leaky_relu', 'hard_tanh', 'hard_sigmoid', 'selu', 'celu', 'log_sigmoid', 'sin', 'cos']
    
    @classmethod 
    def list_initializers(cls) -> list:
        """List all available initializers."""
        return ['glorot_normal', 'glorot_uniform', 'normal', 'he_normal', 
                'he_uniform', 'xavier_normal', 'xavier_uniform', 'zeros', 'ones']
    
    def setup(self):
        config = self.config
        
        init_fn = self.get_initializer(config.init_fn)
        activation = self.get_activation(config.activation)

        if config.arch == 'PirateNet':
            self.model = PirateNet(num_blocks=config.num_layers,
                              embed_scale=config.embed_scale,
                              embed_dim=config.embed_dim,
                              hidden_dim=config.hidden_dim,
                              out_dim=config.out_dim,
                              act=activation,
                              init_fn=init_fn,
                              factorized=config.fact_weight)
        
        elif config.arch == 'MLP':
            self.model = MLP(num_layers=config.num_layers,
                        hidden_dim=config.hidden_dim,
                        out_dim=config.out_dim,
                        act=activation,
                        init_fn=init_fn,
                        factorized=config.fact_weight)
        
        else:
            raise NotImplementedError(f"Arch {config.arch} is not supported!")
    
    @compact
    def __call__(self, x):
        return self.model(x)


class ReBaNO(Module):
    """Reduced Basis Neural Operator using pre-trained PINNs as basis functions."""
    ckpt_data: List[Any]
    c_initial: Any
    
    def setup(self):
        
        num_basis = len(self.ckpt_data)
        self.coefficients = self.param('coefficients', 
                                     lambda rng, shape: self.c_initial, 
                                     (num_basis))
        pinns = []
        pinn_params = []
        
        for ckpt in self.ckpt_data:
            pinn_config_dict = ckpt['metadata']['pinn_config']
            pinn_config = ConfigDict(pinn_config_dict)
            pinn_model = PINN(config=pinn_config)
            pinns.append(pinn_model)
            pinn_params.append(ckpt['params'])
        
        self.pinns = tuple(pinns)
        self.pinn_params = tuple(pinn_params)
    
    def evaluate_pinns(self, x):
        """Evaluate all pre-trained PINNs at single input point."""
        
        pinn_outputs = []
        for pinn, pinn_params in zip(self.pinns, self.pinn_params):
            input = x[None, :]
            output = pinn.apply(pinn_params, input)
            result = output[0] if output.ndim > 1 else output
            pinn_outputs.append(result)
        
        pinn_vals = jnp.stack(pinn_outputs, axis=0)
        return pinn_vals
    
    def __call__(self, x):
        """Forward pass: output = Σᵢ cᵢ * u_pinn^i(x)"""
        pinn_outputs = self.evaluate_pinns(x)  
        return jnp.einsum('i,i...->...', self.coefficients, pinn_outputs)