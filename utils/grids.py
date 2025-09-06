import numpy as np 
import jax.numpy as jnp 
from scipy.special import roots_legendre

def spatial_grid1d(Xi: float=0.0, 
                   Xf: float=1.0, 
                   Nx: int=100, 
                   sampling_method: str='uniform',
                   endpoint: bool=False):
    """ 
        Discretize the 1-dimensional spatial domain with uniform nodes or Chebyshev nodes.
        Return an array containing the coordinates of collocation points (x_i) and quadrature weights.
        Return shape: coord: (Nx, 1), weights: (Nx, 1).
    """
    assert Nx > 1 
    if sampling_method == 'uniform':
        xx = jnp.linspace(Xi, Xf, Nx, endpoint=endpoint)

        coords = xx[:, None]
        if endpoint:
            weights = (Xf - Xi) * jnp.full((Nx, 1), 1.0 / (Nx - 1))
        else:
            weights = (Xf - Xi) * jnp.full((Nx, 1), 1.0 / Nx)

        return coords, weights

    elif sampling_method == 'chebyshev':
        nx = jnp.arange(Nx)
        if endpoint:
            xx = jnp.flip(jnp.cos(nx*jnp.pi/(Nx-1)))
            weights = jnp.ones(Nx)
            weights = weights.at[0].set(0.5)
            weights = weights.at[-1].set(0.5)
            weights = weights * jnp.pi / (Nx - 1)
        else:
            xx = jnp.flip(jnp.cos((nx+0.5)*jnp.pi/Nx))
            weights = jnp.full(Nx, jnp.pi/Nx)
            
        coords = xx[:,None]
        coords = 0.5 * (Xf - Xi) * (coords + 1.0) + Xi
        
        weights = weights[:, None] * 0.5 * (Xf - Xi)
        
        return coords, weights
    
    elif sampling_method == 'gauss':
        assert endpoint == False, "Gauss nodes do not include endpoints."
        nodes, weights = roots_legendre(Nx)
        xx = 0.5 * (Xf - Xi) * (nodes + 1.0) + Xi
        coords = jnp.array(xx)[:, None]
        weights = jnp.array(weights)[:, None] * 0.5 * (Xf - Xi)
    
    else:
        raise NotImplementedError(f"Sampling method '{sampling_method}' is not supported! Supported methods are: uniform, chebyshev, gauss")
         
    return coords, weights

def spatial_grid2d(Xi: float=0.0, 
                   Xf: float=1.0, 
                   Yi: float=0.0, 
                   Yf: float=1.0, 
                   Nx: int=100, 
                   Ny: int=100, 
                   sampling_method: str='uniform',
                   endpoint: bool=False):
    """ 
        Discretize the 2-dimensional spatial domain with uniform nodes or Chebyshev nodes.
        Return an array containing the coordinates of collocation points (x_i, y_j) and an array of quadrature weights.
        Return shape: coords: (Nx*Ny, 2), weights: (Nx*Ny, 1).
    """
    xx, x_weights = spatial_grid1d(Xi, Xf, Nx, 
                                sampling_method=sampling_method,
                                endpoint=endpoint)
    yy, y_weights = spatial_grid1d(Yi, Yf, Ny, 
                                sampling_method=sampling_method,
                                endpoint=endpoint)
    
    x_mesh, y_mesh = jnp.meshgrid(xx[:, 0], yy[:, 0], indexing='xy')
    x_mesh, y_mesh = x_mesh.flatten(), y_mesh.flatten()
    
    weights_x, weights_y = jnp.meshgrid(x_weights[:, 0], y_weights[:, 0], indexing='xy')
    weights_x, weights_y = weights_x.flatten(), weights_y.flatten()
    
    coords = jnp.stack([x_mesh, y_mesh], axis=1)
    weights = (weights_x * weights_y)[:, None]

    return coords, weights

def spatiotemp_grid1d(Xi: float=0.0, 
                      Xf: float=1.0, 
                      Ti: float=0.0,
                      Tf: float=10.0,
                      Nx: int=100, 
                      Nt: int=100,
                      sampling_method: str='uniform',
                      endpoint: bool=False):
    """ 
        Discretize the 1+1 spatio-temporal domain with uniform nodes or Chebyshev nodes in spatial dimensions.
        Return an array containing the coordinates of collocation points (x_i, t_k) and an array of quadrature weights.
        Return shape: coords: (Nx*(Nt+1), 2), weights: (Nx*(Nt+1), 1).
    """
    xx, x_weights = spatial_grid1d(Xi, Xf, Nx, 
                                sampling_method=sampling_method,
                                endpoint=endpoint)
    
    x_coords  = jnp.broadcast_to(xx, (Nt+1, Nx, 1))
    x_weights = jnp.broadcast_to(weights, (Nt+1, Nx, 1))

    t_span   = jnp.linspace(Ti, Tf, Nt+1, endpoint=True)
    t_coords = jnp.broadcast_to(t_span[:, None], (Nt+1, Nx, 1))
    
    coords = jnp.concatenate([x_coords.reshape(-1, 1), 
                              t_coords.reshape(-1, 1)], axis=1)
    weights = x_weights.reshape(-1, 1)
        
    return coords, weights

def spatiotemp_grid2d(Xi: float=0.0, 
                      Xf: float=1.0, 
                      Yi: float=0.0, 
                      Yf: float=1.0, 
                      Ti: float=0.0,
                      Tf: float=10.0,
                      Nx: int=100, 
                      Ny: int=100, 
                      Nt: int=100,
                      sampling_method: str='uniform',
                      endpoint: bool=False):
    """ 
        Discretize the 2+1 spatio-temporal domain with uniform nodes or Chebyshev nodes in spatial dimensions.
        Return an array containing the coordinates of collocation points (x_i, y_j, t_k) and an array of quadrature weights.
        Return shape: coords: (Nx*Ny*(Nt+1), 3), weights: (Nx*Ny*(Nt+1), 1).
    """
    Nxy = Nx * Ny
    xy_coords, xy_weights = spatial_grid2d(Xi, Xf, Yi, Yf, Nx, Ny,
                               sampling_method=sampling_method,
                               endpoint=endpoint)
    xy_coords  = jnp.broadcast_to(xy_coords, (Nt+1, Nxy, 2))
    xy_weights = jnp.broadcast_to(xy_weights, (Nt+1, Nxy, 1))
    
    t_span   = jnp.linspace(Ti, Tf, Nt+1, endpoint=True)
    t_coords = jnp.broadcast_to(t_span[:, None], (Nt+1, Nxy, 1))
    
    coords = jnp.concatenate([xy_coords.reshape(-1, 2), 
                              t_coords.reshape(-1, 1)], axis=1)
    weights = xy_weights.reshape(-1, 1)

    return coords, weights

def spatial_grid1d_bc(Xi: float=0.0, 
                      Xf: float=1.0):
    """ 
        Returns a tuple of the collocation points on the boundary (1d):
                (x_left, x_right).
        and corresponding weights.
        Return shape: coords: (2, 1), weights: (2, 1).
    """
    coords = jnp.array([[Xi],[Xf]])
    weights = jnp.array([[0.5],[0.5]])
    return coords, weights

def spatial_grid2d_bc(Xi: float=0.0,
                      Xf: float=1.0,
                      Yi: float=0.0,
                      Yf: float=1.0,
                      Nx: int=100,
                      Ny: int=100,
                      sampling_method: str='uniform',
                      endpoint: bool=True):
    """ 
        Returns a tuple of the collocation points on the boundary (2d):
                (xy_bottom, xy_right, xy_top, xy_left).
        and corresponding weights.
        Return shape: (N, 2) for each coordinate array and (N, 1) for each weight array where N is Nx or Ny depending on the boundary.
    """
    x_coords, x_weights = spatial_grid1d(Xi, Xf, Nx, sampling_method, endpoint=endpoint)
    y_coords, y_weights = spatial_grid1d(Yi, Yf, Ny, sampling_method, endpoint=endpoint)
    
    x_coords = x_coords.squeeze(-1)
    y_coords = y_coords.squeeze(-1)

    xy_bottom = jnp.stack([x_coords, jnp.full_like(x_coords, Yi)], axis=1)
    xy_top    = jnp.stack([x_coords, jnp.full_like(x_coords, Yf)], axis=1)
    xy_left   = jnp.stack([jnp.full_like(y_coords, Xi), y_coords], axis=1)
    xy_right  = jnp.stack([jnp.full_like(y_coords, Xf), y_coords], axis=1)
    
    coords  = (xy_bottom, xy_right, xy_top, xy_left)
    weights = (x_weights, y_weights, x_weights, y_weights)
    
    return coords, weights

def spatiotemp_grid1d_bc(Xi: float=0.0, 
                         Xf: float=1.0,
                         Ti: float=0.0,
                         Tf: float=10.0,
                         Nt: int=100):
    """ 
        Returns a tuple of the collocation points on the boundary (1+1d):
                (xt_left, xt_right).
        and corresponding weights.
        Return shape: (Nt+1, 2) for each coordinate array and (Nt+1, 1) for each weight array.
    """
    x_left  = jnp.array([[Xi]])
    x_right = jnp.array([[Xf]])
    weight_left = jnp.array([[0.5]])
    weight_right = jnp.array([[0.5]])

    t_coords = jnp.linspace(Ti, Tf, Nt+1, endpoint=True)
    
    x_left  = jnp.broadcast_to(x_left, (Nt+1, 1))
    x_right = jnp.broadcast_to(x_right, (Nt+1, 1))
    weights_left  = jnp.broadcast_to(weight_left, (Nt+1, 1))
    weights_right = jnp.broadcast_to(weight_right, (Nt+1, 1))
    
    xt_left  = jnp.concatenate([x_left, t_coords], axis=1)
    xt_right = jnp.concatenate([x_right, t_coords], axis=1)
    
    coords  = (xt_left, xt_right)
    weights = (weights_left, weights_right)
    
    return coords, weights

def spatiotemp_grid2d_bc(Xi: float=0.0, 
                         Xf: float=1.0,
                         Yi: float=0.0,
                         Yf: float=1.0,
                         Ti: float=0.0,
                         Tf: float=10.0,
                         Nx: int=100, 
                         Ny: int=100, 
                         Nt: int=100,
                         sampling_method: str='uniform',
                         endpoint: bool=True):
    """ 
        Returns a tuple of the collocation points on the boundary (2+1d):
                (xyt_bottom, xyt_right, xyt_top, xyt_left).
        and corresponding weights.
        Return shape: ((Nt+1)*N, 3) for each coordinate array and ((Nt+1)*N, 1) for each weight array where N is Nx or Ny depending on the boundary.
    """
    xy_bc, weights_bc = spatial_grid2d_bc(Xi, Xf, Yi, Yf, Nx, Ny,
                              sampling_method=sampling_method, endpoint=endpoint)
    
    xy_bottom, xy_right, xy_top, xy_left = xy_bc
    weights_bottom, weights_right, weights_top, weights_left = weights_bc

    xy_bottom = jnp.broadcast_to(xy_bottom, (Nt+1, Nx, 2))
    xy_right  = jnp.broadcast_to(xy_right, (Nt+1, Ny, 2))
    xy_top    = jnp.broadcast_to(xy_top, (Nt+1, Nx, 2))
    xy_left   = jnp.broadcast_to(xy_left, (Nt+1, Ny, 2))
    
    weights_bottom = jnp.broadcast_to(weights_bottom, (Nt+1, Nx, 1))
    weights_right  = jnp.broadcast_to(weights_right, (Nt+1, Ny, 1))
    weights_top    = jnp.broadcast_to(weights_top, (Nt+1, Nx, 1))
    weights_left   = jnp.broadcast_to(weights_left, (Nt+1, Ny, 1))

    t_coords = jnp.linspace(Ti, Tf, Nt+1, endpoint=True)
    t_coords_bt = jnp.broadcast_to(t_coords[:, None], (Nt+1, Nx, 1))
    t_coords_lr = jnp.broadcast_to(t_coords[:, None], (Nt+1, Ny, 1))
    
    xyt_bottom = jnp.concatenate([xy_bottom.reshape(-1, 2), 
                                  t_coords_bt.reshape(-1, 1)], axis=1)
    xyt_right  = jnp.concatenate([xy_right.reshape(-1, 2), 
                                  t_coords_lr.reshape(-1, 1)], axis=1)
    xyt_top    = jnp.concatenate([xy_top.reshape(-1, 2), 
                                  t_coords_bt.reshape(-1, 1)], axis=1)
    xyt_left   = jnp.concatenate([xy_left.reshape(-1, 2), 
                                  t_coords_lr.reshape(-1, 1)], axis=1)

    coords = (xyt_bottom, xyt_right, xyt_top, xyt_left)
    weights = (weights_bottom, weights_right, weights_top, weights_left)

    return coords, weights

def spatiotemp_grid1d_ic(Xi: float=0.0, 
                         Xf: float=1.0,
                         T0: float=0.0,
                         Nx: int=100,
                         sampling_method: str='uniform',
                         endpoint: bool=True):
    """ 
        Returns the collocation points at the initial time (1d):
                (x_i, t0) and quadrature weights.
        Return shape: coords: (Nx, 2), weights: (Nx, 1).
    """
    x_coords, x_weights = spatial_grid1d(Xi, Xf, Nx, 
                              sampling_method=sampling_method,
                              endpoint=endpoint)
    
    t0_coords = jnp.full((x_coords.shape[0], 1), T0)
    xt0 = jnp.concatenate([x_coords, t0_coords], axis=1)
    
    return xt0, x_weights

def spatiotemp_grid2d_ic(Xi: float=0.0, 
                         Xf: float=1.0,
                         Yi: float=0.0,
                         Yf: float=1.0,
                         T0: float=0.0,
                         Nx: int=100, 
                         Ny: int=100, 
                         sampling_method: str='uniform',
                         endpoint: bool=True):
    """ 
        Returns the collocation points at the initial time (2d):
                (x_i, y_j, t0) and quadrature weights.
        Return shape: coords: (Nx*Ny, 3), weights: (Nx*Ny, 1).
    """
    xy_coords, xy_weights = spatial_grid2d(Xi, Xf, Yi, Yf, Nx, Ny,
                               sampling_method=sampling_method,
                               endpoint=endpoint)
    
    t0_full = jnp.full((xy_coords.shape[0], 1), T0)
    xyt0    = jnp.concatenate([xy_coords, t0_full], axis=1)
    
    return xyt0, xy_weights
