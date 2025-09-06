import numpy as np 
import jax 
import jax.numpy as jnp
from jax import vmap, jit
from typing import Any, Callable, Dict
from functools import partial
import sys
import os

try:
    from ..utils.utilities import get_u
    from ..utils.gradients import laplacian_vector
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.utilities import get_u
    from utils.gradients import laplacian_vector

Array = jax.Array

@partial(jit, static_argnames=['apply_fn'])
def poisson_residual_loss(apply_fn: Callable,
                          params: Any,
                          col_points: Array,
                          quad_weights: Array,
                          loss_data: Any) -> float:
    """ Get the residual loss of Poisson problem: - Δu = f; L = || - Δu - f ||_2^2 """

    f = loss_data['f']

    lap_u = vmap(partial(laplacian_vector, apply_fn, params))(col_points)

    lossr = jnp.sum(jnp.sum(quad_weights*(lap_u + f)**2, axis=0))
    
    return lossr

@partial(jit, static_argnames=['apply_fn'])
def poisson_boundary_loss(apply_fn: Callable,
                          params: Any,
                          col_points: Array,
                          quad_weights: Array) -> float:
    """ Get the boundary loss of Poisson problem with homogeneous Dirichlet BC """
    
    u_fn = get_u(apply_fn)
    u_bc = vmap(partial(u_fn, params))(col_points)

    lossb = jnp.sum(jnp.sum(quad_weights*u_bc**2, axis=0))

    return lossb

@jit
def poisson_residual_loss_precomp(params: Any,
                                  quad_weights: Array,
                                   u_xx: Array,
                                   loss_data: Any):
    """ Precompute the residual loss for Poisson problem. """

    f = loss_data['f']
    coef = params['params']['coefficients']
    
    u_xx_rb = jnp.einsum('i,ijk->jk', coef, u_xx)

    lossr = jnp.sum(jnp.sum(quad_weights*(u_xx_rb + f)**2, axis=0))

    return lossr

@jit
def poisson_boundary_loss_precomp(params: Any,
                                  quad_weights: Array,
                                   u_bc: Array):
    """ Precompute the boundary loss for Poisson problem with homogeneous Dirichlet BC """
    
    coef = params['params']['coefficients']
    u_bc = jnp.einsum('i,ijk->jk', coef, u_bc)

    lossb = jnp.sum(jnp.sum(quad_weights*u_bc**2, axis=0))

    return lossb

@jit
def poisson_residual_loss_grad(params: Any,
                               quad_weights: Array,
                               u_xx: Array,
                               loss_data: Any):
    """ Precompute the gradient of the residual loss for Poisson problem. """

    f = loss_data['f'][None, ...]
    quad_weights = quad_weights[None, ...]
    coef = params['params']['coefficients']

    u_xx_rb = jnp.einsum('i,ijk->jk', coef, u_xx)[None, ...]

    grad = 2 * jnp.sum(jnp.sum(quad_weights*(u_xx_rb + f) * u_xx, axis=1), axis=1)

    return {'params': {'coefficients': grad}}

@jit
def poisson_boundary_loss_grad(params: Any,
                               quad_weights: Array,
                               u_bc: Array):
    """ Precompute the gradient of the boundary loss for Poisson problem with homogeneous Dirichlet BC """
    
    coef = params['params']['coefficients']
    quad_weights = quad_weights[None, ...]
    u_bc_rb = jnp.einsum('i,ijk->jk', coef, u_bc)[None, ...]

    grad = 2 * jnp.sum(jnp.sum(quad_weights*u_bc_rb * u_bc, axis=1), axis=1)

    return {'params': {'coefficients': grad}}
