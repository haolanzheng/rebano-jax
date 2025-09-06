import jax 
import jax.numpy as jnp 

from jax import vmap, grad, jvp, jacrev 

from typing import Any, Callable

from .utilities import get_u

Array = jax.Array

def grad_u(apply_fn: Callable, params: Any, x: Array):
    """
    Precompute the gradient of u.
    Return shape (out_dim, x_dim).
    """
    u_fn = get_u(apply_fn)
    grad_u_vals = jacrev(u_fn, argnums=1)(params, x)
    return grad_u_vals

def hessian(apply_fn: Callable, params: Array, x: Array):
    """ 
    Compute the hessian for scalar-valued functions.
    Return shape (d, d) where d is x_dim.
    """
    u_fn = get_u(apply_fn)
    def scalar_u(params, xy):
        result = u_fn(params, xy)
        if result.ndim == 0:
            return result
        elif result.ndim == 1 and result.shape[0] == 1:
            return result[0]
        else:
            raise ValueError(f"hessian expects scalar output, got shape {result.shape}")
    
    du = grad(lambda z: scalar_u(params, z))
    d  = x.shape[-1]
    Id = jnp.eye(d)
    
    def hvp(e: Array):
        return jvp(du, (x,), (e,))[1]

    hessian_matrix = vmap(hvp)(Id)
    
    return hessian_matrix

def _laplacian(u_fn: Callable, params: Array, x: Array):
    """
    Compute the Laplacian for scalar-valued functions.
    Return shape scalar.
    """
    def scalar_u(xy):
        result = u_fn(params, xy)
        if result.ndim == 0:
            return result
        elif result.ndim == 1 and result.shape[0] == 1:
            return result[0]
        else:
            raise ValueError(f"laplacian expects scalar output, got shape {result.shape}")
    
    d = x.shape[-1]
    
    def compute_d2udx2(i):
        e = jnp.zeros(d).at[i].set(1.0)
        
        def grad_xi(coords):
            return jvp(scalar_u, (coords,), (e,))[1]
        
        return jvp(grad_xi, (x,), (e,))[1]
    
    d2udx2 = vmap(compute_d2udx2)(jnp.arange(d))
    return jnp.sum(d2udx2)

def laplacian(apply_fn: Callable, params: Array, x: Array, component: int = 0):
    """ 
    Compute the Laplacian for the n-th component of input functions.
    Return shape scalar
    """
    u_fn = get_u(apply_fn)
    def u_n(params, xy):
        result = u_fn(params, xy)
        if result.ndim == 0:
            return result
        elif result.ndim == 1:
            return result[component]
        else:
            raise ValueError(f"Unexpected result shape: {result.shape}")
    
    return _laplacian(u_n, params, x)

def laplacian_vector(apply_fn: Callable, params: Array, x: Array, component=None):
    """
    Compute the Laplacian for vector-valued functions at a single point.
        
    Returns:
        If component is specified: scalar laplacian of that component
        Otherwise: array of shape (out_dim,) with laplacian of each component
    """
    u_fn = get_u(apply_fn)
    test_output = u_fn(params, x)
    
    if test_output.ndim == 0:
        return laplacian(apply_fn, params, x, component=0)
    else:
        out_dim = test_output.shape[0]
        
        if component is not None:
            if component >= out_dim:
                raise ValueError(f"Component {component} out of bounds for function with {out_dim} components")
            return laplacian(apply_fn, params, x, component=component)
        else:
            return vmap(lambda i:
                laplacian(apply_fn, params, x, component=i)
            )(jnp.arange(out_dim))

def dudt(apply_fn: Callable, params: Any, xt: Array):
    """
    Precompute the time derivative of u. xt is organized as [..., t].
    Return shape (out_dim,).
    """
    u_fn = get_u(apply_fn)
    time_unit = jnp.zeros_like(xt).at[-1].set(1.0)
    dudt_val = jvp(lambda coords: u_fn(params, coords), (xt,), (time_unit,))[1]
    return dudt_val

def d2udt2(apply_fn: Callable, params: Any, xt: Array):
    """
    Precompute the second time derivative of u. xt is organized as [..., t].
    Return shape (out_dim,).
    """
    time_unit = jnp.zeros_like(xt).at[-1].set(1.0)

    _dudt = lambda coords: dudt(apply_fn, params, coords)
    d2udt2_val = jvp(_dudt, (xt,), (time_unit,))[1]
    
    return d2udt2_val
