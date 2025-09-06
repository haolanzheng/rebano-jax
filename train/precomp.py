import jax.numpy as jnp 
import jax 
from jax import jacrev, jvp, vmap

import sys
import os

from typing import Any, Callable
from functools import partial

try:
    from ..utils.utilities import get_u
    from ..utils.gradients import grad_u, laplacian, laplacian_vector, dudt, d2udt2
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.utilities import get_u
    from utils.gradients import grad_u, laplacian, laplacian_vector, dudt, d2udt2

Array = jax.Array


def compute_grad_u(apply_fn: Callable, params: Any, x: Array):
    """
    Precompute the gradient of u.
    Return shape (N_points, out_dim, x_dim).
    """
    grad_u_vals = vmap(partial(grad_u, apply_fn, params))(x)
    return grad_u_vals

def compute_lap_u_scalar(apply_fn: Callable, params: Any, x: Array):
    """
    Precompute the Laplacian of scalar function u.
    Return shape (N_points,).
    """
    lap_u = vmap(partial(laplacian, apply_fn, params))(x)[..., None]
    return lap_u
    

def compute_lap_u_vector(apply_fn: Callable, params: Any, x: Array, component: int = None):
    """
    Precompute the Laplacian of all components of vector-valued function u.
    Return shape (N_points,) if component is specified otherwise (N_points, out_dim).
    """
    
    compute_laplacians = partial(laplacian_vector, apply_fn=apply_fn, params=params, component=component) 
            
    return vmap(compute_laplacians)(x)
    
def compute_dudt(apply_fn: Callable, params: Any, xt: Array):
    """
    Precompute the time derivative of u. xt is organized as (..., t).
    Return shape (N_points, n_dim).
    """
    return vmap(partial(dudt, apply_fn, params))(xt)

def compute_d2udt2(apply_fn: Callable, params: Any, xt: Array):
    """
    Precompute the second time derivative of u. xt is organized as (..., t).
    Return shape (N_points, n_dim).
    """
    return vmap(partial(d2udt2, apply_fn, params))(xt)


