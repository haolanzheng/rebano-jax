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

def cg_solve(G, b, G_diag=None, tol=1e-12, maxiter=5000):
    
    G_func = G if callable(G) else (lambda x: G @ x)
    diag = G_diag if callable(G) else jnp.diag(G)
    M = lambda z: z / diag
    
    return cg(G_func, b, M=M, tol=tol, maxiter=maxiter)[0]
     

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
                G_diag: float,
                loss_data: Any) -> float:
    """ Compute the variational loss for the Darcy problem: - ∇⋅(a ∇u) = f. The loss is given by L = R^T @ G^-1 @ R, where 
                R[i] := ∫a ∇u ⋅ ∇v[i] dx - ∫fv[i] dx
    and G is the Gram matrix G[m, n] := ⟨v[n], v[m]⟩_{H^1}
    Args:
        apply_fn: PINN apply function;
        params: PINN parameters;
        quad_points: Quadrature points (N_nodes, N_quad, 2);
        quad_weights: Quadrature weights (N_nodes, N_quad);
        test_fns_values: Test functions: Array of test functions {v_i} values at quadrature points (N_nodes, N_quad);
        grad_test_fn_values: Gradients of test functions: Array of gradients of test functions {∇v_i} values at quadrature points (N_nodes, N_quad, xy_dim);
        gram_mat: function to compute the Gram matrix G matvec;
        G_diag: the diagnal preconditioner value for CG solve of Gram matrix system;
        loss_data: Loss data: dict of loss data ['a': a, 'f': f];
    """

    a_data = loss_data['a'] # (N_nodes, N_quad)
    f_data = loss_data['f'] # (N_nodes, N_quad)
    points_flat = quad_points.reshape(-1, quad_points.shape[-1])  # (N_nodes*N_quad, 2)
    grad_flat = vmap(partial(grad_u, apply_fn, params))(points_flat).squeeze(axis=-2) # (N_nodes*N_quad, xy_dim)
    grad_u_vals = grad_flat.reshape(quad_points.shape)  # (N_nodes, N_quad, xy_dim)

    loss_fn = vmap(loss_single_elem, in_axes=0)

    R = loss_fn(quad_weights, a_data, f_data, grad_u_vals, test_fn_values, grad_test_fn_values)

    total_loss = R.T @ cg_solve(gram_mat, R, G_diag)

    return total_loss

def darcy_boundary_loss(apply_fn: Callable,
                        params: Any,
                        x_bc: Array,
                        quad_weights: Array):
    """ Get the boundary loss of Darcy problem with homogeneous Dirichlet BC """
    
    u_fn = get_u(apply_fn)
    
    u_bc = vmap(partial(u_fn, params))(x_bc.reshape(-1, x_bc.shape[-1]))

    loss_b = jnp.sum(quad_weights.reshape(-1, 1)*u_bc**2)

    return loss_b

def darcy_residual_loss_precomp(params: Any,
                        quad_weights: Array,
                        grad_u_precomp: Array,
                        test_fn_values: Array,
                        grad_test_fn_values: Array,
                        gram_mat: Callable,
                        G_diag: float,
                        loss_data: Any) -> float:
    """ 
    Compute the loss with precomputed gradients of u: u = ⫇_i c_iu_i.
    Args:
        grad_u_precomp: Precomputed gradients of u at quadrature points (n_neurons, N_nodes, N_quad, xy_dim);
    """

    a_data = loss_data['a'] # (N_nodes, N_quad)
    f_data = loss_data['f'] # (N_nodes, N_quad)

    grad_u_precomp = jnp.transpose(grad_u_precomp, (1, 0, 2, 3))  # (N_nodes, n_neurons, N_quad, xy_dim)
    # print(grad_u_precomp.shape)
    coef = params['params']['coefficients']
    grad_u_rb = jnp.einsum('i,eijk->ejk', coef, grad_u_precomp)  # (N_nodes, N_quad, xy_dim)
    
    loss_fn = vmap(loss_single_elem, in_axes=0)

    R = loss_fn(quad_weights, a_data, f_data, grad_u_rb, test_fn_values, grad_test_fn_values)

    total_loss = R.T @ cg_solve(gram_mat, R, G_diag)

    return total_loss

def darcy_boundary_loss_precomp(params: Any,
                                u_bc: Array,
                                quad_weights: Array):
    """ Get the boundary loss of Darcy problem with homogeneous Dirichlet BC """
    coef = params['params']['coefficients']
    u_bc = u_bc.reshape(u_bc.shape[0], -1)
    u_bc_rb = jnp.einsum('i,ij->j', coef, u_bc)

    loss_b = jnp.sum(quad_weights.flatten()*u_bc_rb**2)

    return loss_b

def darcy_residual_loss_grad(params: Any,
                        quad_weights: Array,
                        grad_u_precomp: Array,
                        test_fn_values: Array,
                        grad_test_fn_values: Array,
                        gram_mat: Callable,
                        G_diag: float,
                        loss_data: Any) -> float:
    """ 
    Compute the loss with precomputed gradients of u.
    Args:
        grad_u_precomp: Precomputed gradients of u at quadrature points (n_neurons, N_nodes, N_quad, xy_dim);
    """

    a_data = loss_data['a'] # (N_nodes, N_quad)
    f_data = loss_data['f'] # (N_nodes, N_quad)
    
    grad_u_precomp = jnp.transpose(grad_u_precomp, (1, 0, 2, 3))  # (N_nodes, n_neurons, N_quad, xy_dim)
    
    coef = params['params']['coefficients']
    grad_u_rb = jnp.einsum('i,eijk->ejk', coef, grad_u_precomp)  # (N_nodes, N_quad, xy_dim)
    
    grad_loss_fn = vmap(grad_loss_single_elem, in_axes=0)
    
    R = vmap(loss_single_elem, in_axes=0)(quad_weights, a_data, f_data, grad_u_rb, test_fn_values, grad_test_fn_values)

    dR = grad_loss_fn(quad_weights, a_data, grad_u_precomp, grad_test_fn_values)

    grad = 2.0 * dR.T @ cg_solve(gram_mat, R, G_diag)

    return {'params': {'coefficients': grad}}

def darcy_boundary_loss_grad(params: Any,
                             u_bc: Array,
                             quad_weights: Array):
    """ Get the gradient of boundary loss of Darcy problem with homogeneous Dirichlet BC """
    coef = params['params']['coefficients']
    u_bc = u_bc.reshape(u_bc.shape[0], -1)
    u_bc_rb = jnp.einsum('i,ij->j', coef, u_bc)

    grad = 2.0 * jnp.sum(quad_weights.reshape(1, -1) * u_bc_rb[None, :] * u_bc)

    return {'params': {'coefficients': grad}}
