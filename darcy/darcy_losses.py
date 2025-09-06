import numpy as np 
import jax 
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.sparse.linalg import cg
from typing import Any, Callable, Dict
from functools import partial
import sys
import os

try:
    from ..utils.utilities import get_u
    from ..utils.gradients import grad_u
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.utilities import get_u
    from utils.gradients import grad_u

Array = jax.Array

def cg_solve(G, b, v1, tol=1e-6, maxiter=1000):
    
    G = G if callable(G) else lambda x: G @ x

    M = lambda z: z / v1
    
    x = cg(G, b, M=M, tol=tol, maxiter=maxiter)[0]
    
    return x
     

def loss_single_elem(quad_w_elem: Array,
                    a_values_elem: Array,
                    f_values_elem: Array,
                    grad_u_values_elem: Array,
                    test_fn_values_elem: Array,
                    grad_test_fn_values_elem: Array):
    """ Compute the loss of a single element. """

    loss = jnp.sum(
        quad_w_elem * a_values_elem * jnp.sum(grad_u_values_elem * grad_test_fn_values_elem, axis=-1) - 
        quad_w_elem * f_values_elem * test_fn_values_elem
    )

    return loss

def grad_loss_single_elem(quad_w_elem: Array,
                    a_values_elem: Array,
                    grad_u_values_elem: Array,
                    grad_test_fn_values_elem: Array):
    """ Compute the gradient w.r.t. coefficients of the loss from a single element. """
    
    grad = jnp.sum(
        quad_w_elem[None, ...] * a_values_elem[None, ...] * jnp.sum(grad_u_values_elem * grad_test_fn_values_elem[None, ...], axis=-1) , axis=-1
    )

    return grad

def darcy_residual_loss(apply_fn: Callable,
                params: Any,
                quad_points: Array,
                quad_weights: Array,
                test_fn_values: Array,
                grad_test_fn_values: Array,
                gram_mat: Callable,
                v1: float,
                loss_data: Any) -> float:
    """ Compute the variational loss for the Darcy problem: - ∇⋅(a ∇u) = f. The loss is given by L = R^T @ G^-1 @ R, where 
                R[i] := ∫a ∇u ⋅ ∇v[i] dx - ∫fv[i] dx
    and G is the Gram matrix G[m, n] := ⟨v[n], v[m]⟩_{H^1}
    Args:
        apply_fn: PINN apply function;
        params: PINN parameters;
        quad_points: Quadrature points (N_elem, N_quad, 2);
        quad_weights: Quadrature weights (N_elem, N_quad);
        test_fns_values: Test functions: Array of test functions {v_i} values at quadrature points (N_elem, N_quad);
        grad_test_fn_values: Gradients of test functions: Array of gradients of test functions {∇v_i} values at quadrature points (N_elem, N_quad, xy_dim);
        gram_mat: function to compute the Gram matrix G matvec;
        v1: the diagnal preconditioner value for CG solve of Gram matrix system;
        loss_data: Loss data: dict of loss data ['a': a, 'f': f];
    """

    a_data = loss_data['a'] # (N_elem, N_quad)
    f_data = loss_data['f'] # (N_elem, N_quad)
    points_flat = quad_points.reshape(-1, quad_points.shape[-1])  # (N_elem*N_quad, 2)
    grad_flat = vmap(partial(grad_u, apply_fn, params))(points_flat).squeeze(axis=-2) # (N_elem*N_quad, xy_dim)
    grad_u_vals = grad_flat.reshape(quad_points.shape[0], quad_points.shape[1], -1)  # (N_elem, N_quad, xy_dim)

    loss_fn = vmap(loss_single_elem, in_axes=0)

    R = loss_fn(quad_weights, a_data, f_data, grad_u_vals, test_fn_values, grad_test_fn_values)

    total_loss = R.T @ cg_solve(gram_mat, R, v1)

    return total_loss

def darcy_boundary_loss(apply_fn: Callable,
                        params: Any,
                        x_bc: Array,
                        quad_weights: Array):
    """ Get the boundary loss of Darcy problem with homogeneous Dirichlet BC """
    
    u_fn = get_u(apply_fn)
    u_bc_b = vmap(partial(u_fn, params))(x_bc[0])
    u_bc_r = vmap(partial(u_fn, params))(x_bc[1])
    u_bc_t = vmap(partial(u_fn, params))(x_bc[2])
    u_bc_l = vmap(partial(u_fn, params))(x_bc[3])
    
    loss_b = jnp.sum(quad_weights[0]*u_bc_b**2) + jnp.sum(quad_weights[1]*u_bc_r**2) + jnp.sum(quad_weights[2]*u_bc_t**2) + jnp.sum(quad_weights[3]*u_bc_l**2)

    return loss_b

def darcy_residual_loss_precomp(params: Any,
                        quad_weights: Array,
                        grad_u_precomp: Array,
                        test_fn_values: Array,
                        grad_test_fn_values: Array,
                        gram_mat: Callable,
                        v1: float,
                        loss_data: Any) -> float:
    """ 
    Compute the loss with precomputed gradients of u: u = ⫇_i c_iu_i.
    Args:
        grad_u_precomp: Precomputed gradients of u at quadrature points (n_neurons, N_elem, N_quad, xy_dim);
    """

    a_data = loss_data['a'] # (N_elem, N_quad)
    f_data = loss_data['f'] # (N_elem, N_quad)

    grad_u_precomp = jnp.transpose(grad_u_precomp, (1, 0, 2, 3))  # (N_elem, n_neurons, N_quad, xy_dim)
    # print(grad_u_precomp.shape)
    coef = params['params']['coefficients']
    grad_u_rb = jnp.einsum('i,eijk->ejk', coef, grad_u_precomp)  # (N_elem, N_quad, xy_dim)
    
    loss_fn = vmap(loss_single_elem, in_axes=0)

    R = loss_fn(quad_weights, a_data, f_data, grad_u_rb, test_fn_values, grad_test_fn_values)

    total_loss = R.T @ cg_solve(gram_mat, R, v1)

    return total_loss

def darcy_boundary_loss_precomp(params: Any,
                                quad_weights: Array,
                                u_bc: Array):
    """ Get the boundary loss of Darcy problem with homogeneous Dirichlet BC """
    coef = params['params']['coefficients']
    u_bc = u_bc.transpose((1, 0, 2))
    u_bc_rb_b = jnp.einsum('i,ij->j', coef, u_bc[0])
    u_bc_rb_r = jnp.einsum('i,ij->j', coef, u_bc[1])
    u_bc_rb_t = jnp.einsum('i,ij->j', coef, u_bc[2])
    u_bc_rb_l = jnp.einsum('i,ij->j', coef, u_bc[3])

    loss_b = jnp.sum(quad_weights[0]*u_bc_rb_b**2) + jnp.sum(quad_weights[1]*u_bc_rb_r**2) + jnp.sum(quad_weights[2]*u_bc_rb_t**2) + jnp.sum(quad_weights[3]*u_bc_rb_l**2)

    return loss_b

def darcy_residual_loss_grad(params: Any,
                        quad_weights: Array,
                        grad_u_precomp: Array,
                        test_fn_values: Array,
                        grad_test_fn_values: Array,
                        gram_mat: Callable,
                        v1: float,
                        loss_data: Any) -> float:
    """ 
    Compute the loss with precomputed gradients of u.
    Args:
        grad_u_precomp: Precomputed gradients of u at quadrature points (n_neurons, N_elem, N_quad, xy_dim);
    """

    a_data = loss_data['a'] # (N_elem, N_quad)
    f_data = loss_data['f'] # (N_elem, N_quad)
    
    grad_u_precomp = jnp.transpose(grad_u_precomp, (1, 0, 2, 3))  # (N_elem, n_neurons, N_quad, xy_dim)
    
    coef = params['params']['coefficients']
    grad_u_rb = jnp.einsum('i,eijk->ejk', coef, grad_u_precomp)  # (N_elem, N_quad, xy_dim)
    
    grad_loss_fn = vmap(grad_loss_single_elem, in_axes=0)
    
    R = vmap(loss_single_elem, in_axes=0)(quad_weights, a_data, f_data, grad_u_rb, test_fn_values, grad_test_fn_values)

    dR = grad_loss_fn(quad_weights, a_data, grad_u_precomp, grad_test_fn_values)

    grad = 2.0 * dR.T @ cg_solve(gram_mat, R, v1)

    return {'params': {'coefficients': grad}}

def darcy_boundary_loss_grad(params: Any,
                             quad_weights: Array,
                                u_bc: Array):
    """ Get the gradient of boundary loss of Darcy problem with homogeneous Dirichlet BC """
    coef = params['params']['coefficients']
    u_bc = u_bc.transpose((1, 0, 2))
    u_bc_rb_b = jnp.einsum('i,ij->j', coef, u_bc[0])
    u_bc_rb_r = jnp.einsum('i,ij->j', coef, u_bc[1])
    u_bc_rb_t = jnp.einsum('i,ij->j', coef, u_bc[2])
    u_bc_rb_l = jnp.einsum('i,ij->j', coef, u_bc[3])

    grad = 2.0 * jnp.sum(quad_weights[0]*u_bc_rb_b[None, :] * u_bc[0], axis=-1) + \
           2.0 * jnp.sum(quad_weights[1]*u_bc_rb_r[None, :] * u_bc[1], axis=-1) + \
           2.0 * jnp.sum(quad_weights[2]*u_bc_rb_t[None, :] * u_bc[2], axis=-1) + \
           2.0 * jnp.sum(quad_weights[3]*u_bc_rb_l[None, :] * u_bc[3], axis=-1)

    return {'params': {'coefficients': grad}}
