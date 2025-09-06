import jax
import jax.numpy as jnp
from jax import vmap, jit

from typing import Any
import sys, os

try:
    from .grids import spatial_grid1d, spatial_grid2d
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from grids import spatial_grid1d, spatial_grid2d

Array = jax.Array

def test_fn_linear1d(x_start, x_curr, x_end, quad_x):
    """ Return the linear test function values and gradients in 1D. """
    
    quad_x = quad_x.flatten()
    
    cond1 = (quad_x >= x_start) & (quad_x <= x_curr)
    val1 = (quad_x - x_start) / (x_curr - x_start)
    
    cond2 = (quad_x > x_curr) & (quad_x <= x_end)
    val2 = (x_end - quad_x) / (x_end - x_curr)
    
    result = jnp.where(cond1, val1, jnp.where(cond2, val2, 0.0))
    
    return result

def grad_test_fn_linear1d(x_start, x_curr, x_end, quad_x):
    """ Return the gradient of linear test function in 1D. """
    
    quad_x = quad_x.flatten()
    
    cond1 = (quad_x >= x_start) & (quad_x <= x_curr)
    val1 = 1.0 / (x_curr - x_start)
    
    cond2 = (quad_x > x_curr) & (quad_x <= x_end)
    val2 = -1.0 / (x_end - x_curr)
    
    result = jnp.where(cond1, val1, jnp.where(cond2, val2, 0.0))
    
    return result

def test_fn_linear2d(x_start, x_curr, x_end, y_start, y_curr, y_end, quad_xy):
    """ Return the linear test function values in 2D. """

    test_fnx = test_fn_linear1d(x_start, x_curr, x_end, quad_xy[:, 0:1])
    test_fny = test_fn_linear1d(y_start, y_curr, y_end, quad_xy[:, 1:2])
    
    return test_fnx * test_fny

def grad_test_fn_linear2d(x_start, x_curr, x_end, y_start, y_curr, y_end, quad_xy):
    """ Return the gradient of linear test function in 2D. """

    test_fnx = test_fn_linear1d(x_start, x_curr, x_end, quad_xy[:, 0:1])
    test_fny = test_fn_linear1d(y_start, y_curr, y_end, quad_xy[:, 1:2])
    
    grad_test_fnx = grad_test_fn_linear1d(x_start, x_curr, x_end, quad_xy[:, 0:1])
    grad_test_fny = grad_test_fn_linear1d(y_start, y_curr, y_end, quad_xy[:, 1:2])
    
    grad_x = grad_test_fnx * test_fny
    grad_y = test_fnx * grad_test_fny
    
    return jnp.hstack([grad_x[:, None], grad_y[:, None]])

def gram_coef(dx, dy):
    v1 = 4*dx*dy/9 + 4*dy/(3*dx) + 4*dx/(3*dy)
    v2 = dx*dy/9 + dy/(3*dx) - 2*dx/(3*dy)
    v3 = dx*dy/9 - 2*dy/(3*dx) + dx/(3*dy)
    v4 = dx*dy/36 - dy/(6*dx) - dx/(6*dy)
    
    return v1, v2, v3, v4

def gram_mat(Nx, Ny, dx, dy):

    v1, v2, v3, v4 = gram_coef(dx, dy)

    def shift(u, di, dj):
        pad = ((max(di,0), max(-di,0)), (max(dj,0), max(-dj,0)))
        v = jnp.pad(u, pad)
        return v[max(-di,0):v.shape[0]-max(di,0),
                 max(-dj,0):v.shape[1]-max(dj,0)]

    def G(x):
        u = x.reshape(Ny, Nx)
        y = (v1*u # center
             + v2*(shift(u, -1, 0)+shift(u, 1, 0)) # N and S neighbors
             + v3*(shift(u, 0, -1)+shift(u, 0, 1)) # W and E neighbors
             + v4*(shift(u, -1, -1)+shift(u, -1, 1)+shift(u, 1, -1)+shift(u, 1, 1))) # NW, NE, SW, SE neighbors
        return y.ravel()
    return jit(G)
    
    
    
    
    