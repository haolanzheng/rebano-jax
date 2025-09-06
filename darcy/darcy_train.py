import os
if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
    os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")  # Use more CPU devices for better parallelization
    
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_log_compiles", False)
from jax import jit, pmap, vmap, devices, local_device_count
from jax.scipy.interpolate import RegularGridInterpolator
from ml_collections import ConfigDict
import sys
from functools import partial
from tqdm import tqdm
import wandb
import time

try:
    from ..models.nets import PINN, ReBaNO
    from .darcy_losses import *
    from ..train.training import make_step, create_train_state, make_step_rebano
    from ..train.precomp import compute_grad_u
    from ..configs.darcy_config import get_train_config
    from ..utils.grids import spatial_grid1d, spatial_grid2d, spatial_grid2d_bc
    from ..utils.utilities import get_u, save_pinn_checkpoint, load_checkpoint, count_params
    from ..utils.test_fns import *
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.nets import PINN, ReBaNO
    from darcy_losses import *
    from train.training import make_step, create_train_state, make_step_rebano
    from train.precomp import compute_grad_u
    from configs.darcy_config import get_train_config
    from utils.grids import spatial_grid1d, spatial_grid2d, spatial_grid2d_bc
    from utils.utilities import get_u, save_pinn_checkpoint, load_checkpoint, count_params
    from utils.test_fns import *



def train_pinn(state, col_points, quad_weights, test_fn_values, grad_test_fn_values, gram_matrix, v1, loss_weights, alpha,
               max_steps=10000, tol_grad=1e-8, wandb_config=None):
    """Train the PINN model on the Darcy problem."""
    
    def residual_loss_pinn(apply_fn, params, batch_data, quad_weights, loss_data):
        return darcy_residual_loss(apply_fn, params, batch_data, quad_weights, test_fn_values, grad_test_fn_values, gram_matrix, v1, loss_data)
    
    def boundary_loss_pinn(apply_fn, params, batch_data, quad_weights, loss_data=None):
        return darcy_boundary_loss(apply_fn, params, batch_data, quad_weights)

    loss_fns = {'residual': residual_loss_pinn, 'boundary': boundary_loss_pinn}

    step_fn = make_step(loss_fns)
    
    pbar = tqdm(range(max_steps), desc="Training PINN", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for step in pbar:
        state, loss_weights, metrics = step_fn(state, loss_weights,
                                               col_points, quad_weights, alpha)
        
        pbar.set_postfix({
            'Total loss': f"{metrics['total_loss']:.2e}"
        })
        
        if wandb_config and wandb_config.enabled and step % wandb_config.log_freq == 0:
            log_data = {
                'step': step,
                'total_loss': metrics['total_loss'],
                'residual_loss': metrics['residual_loss'],
                'residual_weight': metrics['residual_weight'],
                'boundary_loss': metrics['boundary_loss'],
                'boundary_weight': metrics['boundary_weight'],
            }
            wandb.log(log_data)
        
        if step % 5000 == 0:
            tqdm.write(f"Step {step}: Total={metrics['total_loss']:.5e}, "
                       f"Residual={metrics['residual_loss']:.5e}, BC={metrics['boundary_loss']:.5e}")
        
        if metrics['grad_norm'] < tol_grad:
            tqdm.write(f"Converged at step {step}, final loss: {metrics['total_loss']:.5e}")
            break
        
        if step == max_steps - 1:
            tqdm.write(f"Final loss: {metrics['total_loss']:.5e}")
    
    pbar.close()
    return state

def train_rebano(config_rebano, pinns, col_points, quad_weights, a_data,
                 f_data, test_fn_values, grad_test_fn_values, grad_u_precomp, u_bc_precomp, gram_matrix, v1, available_devices,
                 use_pmap, batch_size, wandb_config=None):
    """Train the ReBaNO on the Poisson problem with pmap parallelization."""
    
    n_devices = len(available_devices)
    if not use_pmap or n_devices == 1:
        print("Using sequential processing")
        return train_rebano_sequential(config_rebano, pinns, 
                                       col_points, quad_weights, a_data,
                                       f_data, test_fn_values, grad_test_fn_values, grad_u_precomp, u_bc_precomp, gram_matrix, v1, wandb_config)

    print(f"Using pmap with {n_devices} devices, batch size {batch_size}")
    return train_rebano_pmap(config_rebano, pinns, col_points, 
                             quad_weights, a_data, f_data, test_fn_values, grad_test_fn_values, grad_u_precomp,  u_bc_precomp, gram_matrix, v1, n_devices, batch_size, available_devices, wandb_config)


def train_rebano_pmap(config_rebano, pinns, col_points, quad_weights,
                      a_data, f_data, test_fn_values, grad_test_fn_values, grad_u_precomp, u_bc_precomp, gram_matrix, v1, n_devices, batch_size, available_devices, wandb_config):
    """Parallel ReBaNO training using pmap."""
    try: 
        max_steps = config_rebano.max_steps
    except AttributeError:
        max_steps = 2000
        
    dummy_key = jax.random.PRNGKey(0)

    n_samples = a_data.shape[-1]
    n_neurons = len(pinns)
    
    def train_single_sample(a_sample):
        c_initial = jnp.full((n_neurons,), 1.0/n_neurons)
        rebano = ReBaNO(pinns, c_initial)
        
        loss_data = {'residual': {'a': a_sample, 'f': f_data}, 'boundary': None}
        loss_weights = {'residual': config_rebano.w_residual, 'boundary': config_rebano.w_bc}
        
        state = create_train_state(dummy_key, rebano, config_rebano,
                                   col_points['residual'], loss_data)

        def residual_loss_rebano(params, quad_weights, loss_data):
            return darcy_residual_loss_precomp(params, quad_weights, grad_u_precomp, test_fn_values, grad_test_fn_values, gram_matrix, v1, loss_data)

        def residual_loss_grad_rebano(params, quad_weights, loss_data):
            return darcy_residual_loss_grad(params, quad_weights, grad_u_precomp, test_fn_values, grad_test_fn_values, gram_matrix, v1, loss_data)
        
        def boundary_loss_rebano(params, quad_weights, loss_data=None):
            return darcy_boundary_loss_precomp(params, u_bc_precomp, quad_weights)

        def boundary_loss_grad_rebano(params, quad_weights, loss_data=None):
            return darcy_boundary_loss_grad(params, u_bc_precomp, quad_weights)

        loss_fns = {'residual': residual_loss_rebano, 'boundary': boundary_loss_rebano}
        grad_fns = {'residual': residual_loss_grad_rebano, 'boundary': boundary_loss_grad_rebano}

        step_fn = make_step_rebano(loss_fns, grad_fns)
        
        def loop_body(step, carry):
            state, loss_weights = carry
            state, loss_weights, metrics = step_fn(state, quad_weights, loss_weights)
            return (state, loss_weights)
        
        # max_steps = 2000
        final_state, final_loss_weights = jax.lax.fori_loop(0, max_steps, loop_body, (state, loss_weights))
        
        # Get final loss
        _, _, final_metrics = step_fn(final_state, quad_weights, final_loss_weights)
        final_loss = final_metrics['true loss']
        
        return final_loss
    
    def train_batch_samples(f_batch):
        batch_losses = vmap(train_single_sample)(f_batch)
        return batch_losses 
    
    pmap_train_batch = pmap(train_batch_samples, axis_name='device',
                            devices=available_devices)
    

    samples_per_device_batch = batch_size
    total_batch_size = samples_per_device_batch * n_devices
    n_batches = (n_samples + total_batch_size - 1) // total_batch_size
    
    global_idx_max = 0
    global_loss_max = 0.0
    processed_samples = 0
    
    print(f"Processing {n_samples} samples in {n_batches} parallel batches")
    print(f"Batch size per device: {samples_per_device_batch}, Total parallel: {total_batch_size}")
    
    pbar_batches = tqdm(range(n_batches), desc="Parallel ReBaNO Training",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx in pbar_batches:
        start_idx = batch_idx * total_batch_size
        end_idx = min(start_idx + total_batch_size, n_samples)
        
        if start_idx >= n_samples:
            break
        
        batch_a_data = prepare_pmap_batch(
            a_data, start_idx, end_idx, samples_per_device_batch, n_devices
        )
        
        try:
            batch_losses = pmap_train_batch(batch_a_data)
            jax.block_until_ready(batch_losses)
            
            flat_losses = batch_losses.flatten()
            
            # Calculate actual number of samples processed (not including padding)
            actual_samples_in_batch = min(end_idx - start_idx, total_batch_size)
            
            valid_mask = jnp.isfinite(flat_losses)
            valid_losses = flat_losses[valid_mask]
            
            if len(valid_losses) > 0:
                batch_max_loss = float(jnp.max(valid_losses))
                batch_max_local_idx = int(jnp.argmax(flat_losses))
                batch_max_global_idx = start_idx + batch_max_local_idx
                
                if batch_max_loss > global_loss_max:
                    global_loss_max = batch_max_loss
                    global_idx_max  = batch_max_global_idx
                
                processed_samples += actual_samples_in_batch
                
                pbar_batches.set_postfix({
                    'Max Loss': f'{global_loss_max:.3e}',
                    'Index max': global_idx_max,
                    'Processed': f'{processed_samples}/{n_samples}'
                })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    pbar_batches.close()
    
    print(f'\nParallel ReBaNO Training Complete!')
    print(f'Processed {processed_samples}/{n_samples} samples')
    print(f'Worst sample: {global_idx_max} with loss {global_loss_max:.5e}')
    
    if wandb_config and wandb_config.enabled:
        log_data = {
            'idx_max': global_idx_max,
            'loss_max': global_loss_max,
            'processed_samples': processed_samples,
            'total_samples': n_samples
        }
        wandb.log(log_data)
    
    return global_idx_max, global_loss_max


def prepare_pmap_batch(a_data, start_idx, end_idx, batch_size, n_devices):
    """Prepare data for pmap - reshape to (n_devices, N_nodes, N_quad, batch_size)"""
    actual_samples = end_idx - start_idx
    total_batch_size = batch_size * n_devices
    
    if actual_samples < total_batch_size:
        a_padded = jnp.zeros((a_data.shape[0], a_data.shape[1], total_batch_size))
        a_padded = a_padded.at[..., :actual_samples].set(a_data[..., start_idx:end_idx])
        if actual_samples > 0:
            last_sample = a_data[..., end_idx-1:end_idx]
            for i in range(actual_samples, total_batch_size):
                a_padded = a_padded.at[..., i:i+1].set(last_sample)
        a_batch = a_padded
    else:
        a_batch = a_data[..., start_idx:start_idx + total_batch_size]

    # a_batch shape: (128, 400) -> (400, 128) -> (4, 100, 128)
    
    a_batch = jnp.transpose(a_batch, (2, 0, 1)).reshape(n_devices, batch_size, a_data.shape[0], a_data.shape[1])

    return a_batch


def train_rebano_sequential(config_rebano, pinns, col_points,
                            quad_weights, a_data, f_data, test_fn_values, grad_test_fn_values, grad_u_precomp, u_bc_precomp, gram_matrix, v1, wandb_config):
    """Sequential ReBaNO training - fallback when pmap is not available."""
    try:
        max_steps = config_rebano.max_steps
        tol_grad = config_rebano.tol_grad
    except AttributeError:
        max_steps = 2000
        tol_grad = 1e-9
        
    def residual_loss_rebano(params, quad_weights, loss_data):
            return darcy_residual_loss_precomp(params, quad_weights, grad_u_precomp, test_fn_values, grad_test_fn_values, gram_matrix, v1, loss_data)

    def residual_loss_grad_rebano(params, quad_weights, loss_data):
            return darcy_residual_loss_grad(params, quad_weights, grad_u_precomp, test_fn_values, grad_test_fn_values, gram_matrix, v1, loss_data)
        
    def boundary_loss_rebano(params, quad_weights, loss_data=None):
            return darcy_boundary_loss_precomp(params, u_bc_precomp, quad_weights)

    def boundary_loss_grad_rebano(params, quad_weights, loss_data=None):
            return darcy_boundary_loss_grad(params, u_bc_precomp, quad_weights)

    n_samples = a_data.shape[-1]
    n_neurons = len(pinns)
    
    dummy_key = jax.random.PRNGKey(0)

    pbar_samples = tqdm(range(n_samples), desc="Sequential ReBaNO Training",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    idx_max = 0
    loss_max = 0.0
    
    loss_fns = {'residual': residual_loss_rebano, 'boundary': boundary_loss_rebano}
    grad_fns = {'residual': residual_loss_grad_rebano, 'boundary': boundary_loss_grad_rebano}

    for sample_idx in pbar_samples:
        a_sample = a_data[..., sample_idx]
        c_initial = jnp.full((n_neurons,), 1.0/n_neurons)
        rebano = ReBaNO(pinns, c_initial)
        
        loss_data = {'residual': {'a': a_sample, 'f': f_data}, 'boundary': None}
        loss_weights = {'residual': config_rebano.w_residual, 'boundary': config_rebano.w_bc}
        
        state = create_train_state(dummy_key, rebano, config_rebano,
                                   col_points['residual'], loss_data)

        step_fn = make_step_rebano(loss_fns, grad_fns)
        
        for step in range(max_steps):
            state, loss_weights, metrics = step_fn(state, quad_weights, loss_weights)
            jax.block_until_ready(metrics['true loss'])
            
            loss_value = metrics['true loss']
            
            if loss_value < loss_max:
                break
            
            if metrics['grad_norm'] < tol_grad:
                idx_max = sample_idx
                loss_max = loss_value
                break
                
            if step == max_steps - 1:
                idx_max = sample_idx 
                loss_max = loss_value
        
        pbar_samples.set_postfix({
            '# of PINNs': n_neurons,
            'maximum loss': f'{loss_max:.5e}',
            'next input function selected index': idx_max,
            'total samples': n_samples,
        })

    pbar_samples.close()
    
    print(f'\nSequential ReBaNO Training Complete!')
    print(f'Best sample: {idx_max} with loss {loss_max:.5e}')
    
    if wandb_config and wandb_config.enabled:
        log_data = {'idx_max': idx_max, 'loss_max': loss_max}
        wandb.log(log_data)
    
    return idx_max, loss_max

def main():
    config = get_train_config()

    # Print device information
    backend = jax.default_backend()
    available_devices = devices('gpu') if backend == 'gpu' else devices('cpu')
    print(f"JAX devices available: {len(available_devices)}")
    for i, device in enumerate(available_devices):
        print(f"  Device {i}: {device}")
    print(f"JAX default backend: {jax.default_backend()}")

    if config.wandb.enabled:
        try:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=f"darcy_rebano_neurons_{config.num_neurons}",
                config=config.to_dict()
            )
            print(f"Wandb initialized successfully for project: {config.wandb.project}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing training without wandb logging...")
            # Disable wandb for this run
            config.wandb.enabled = False

    num_neurons = config.num_neurons

    # loading data
    offset  = config.data.offset
    n_train = config.data.n_samples
    
    inputs  = np.load(config.data.inputs_dir)[::config.data.sub_x, ::config.data.sub_y, offset:n_train+offset].astype(np.float32)
    outputs = np.load(config.data.outputs_dir)[::config.data.sub_x, ::config.data.sub_y, offset:n_train+offset].astype(np.float32)
    
    Ny, Nx = inputs.shape[0], inputs.shape[1]
    
    inputs  = jnp.array(inputs.astype(np.float32))
    outputs = jnp.array(outputs.reshape(-1, n_train).astype(np.float32))
        
    Nelem_x, Nelem_y = config.domain.Nelem_x, config.domain.Nelem_y
    N_nodes = (Nelem_x-1) * (Nelem_y-1) # interior nodes only
    N_quad_1d = config.domain.N_quad  # Number of quadrature points per dimension
    N_quad = N_quad_1d**2  # Total quadrature points per element
    N_bc   = config.domain.N_bc 
    Xi, Xf = config.domain.Xi, config.domain.Xf
    Yi, Yf = config.domain.Yi, config.domain.Yf
    
    # Create test grid matching input spatial dimensions  
    xy_test = spatial_grid2d(Xi, Xf, Yi, Yf, Nx, Ny, 'uniform')[0]

    x_grid, y_grid = spatial_grid1d(Xi, Xf, Nelem_x+1, sampling_method='uniform', endpoint=True)[0], spatial_grid1d(Yi, Yf, Nelem_y+1, sampling_method='uniform', endpoint=True)[0]
    dx, dy = x_grid[1]-x_grid[0], y_grid[1]-y_grid[0]
    
    xy_bc, weights_bc = spatial_grid2d_bc(Xi, Xf, Yi, Yf, N_bc, N_bc, 'uniform')
    xy_bc = jnp.stack([xy_bc[0], xy_bc[1], xy_bc[2], xy_bc[3]], axis=0)
    weights_bc = jnp.stack([weights_bc[0], weights_bc[1], weights_bc[2], weights_bc[3]], axis=0)
    
    xy_bc_flatten = xy_bc.reshape(-1, 2)
    
    # Create interpolation grids matching input data dimensions
    x_interp_grid = spatial_grid1d(Xi, Xf, Nx)[0].flatten()
    y_interp_grid = spatial_grid1d(Yi, Yf, Ny)[0].flatten()
    
    v1 = gram_coef(dx, dy)[0]
    G = gram_mat(Nelem_x-1, Nelem_y-1, dx, dy)

    quad_xy = jnp.zeros((Nelem_y-1, Nelem_x-1, N_quad, 2))
    quadw_xy = jnp.zeros((Nelem_y-1, Nelem_x-1, N_quad))
    test_fn_values = jnp.zeros((Nelem_y-1, Nelem_x-1, N_quad))
    grad_test_fn_values = jnp.zeros((Nelem_y-1, Nelem_x-1, N_quad, 2))

    for j in range(Nelem_x-1):
        for i in range(Nelem_y-1):
            xy_elem, weight_elem = spatial_grid2d(x_grid[j], x_grid[j+2], y_grid[i], y_grid[i+2], N_quad_1d, N_quad_1d, 'gauss')
            
            quad_xy = quad_xy.at[i, j].set(xy_elem)
            quadw_xy = quadw_xy.at[i, j].set(weight_elem.squeeze(-1))
            
            test_fn_elem = test_fn_linear2d(x_grid[j], x_grid[j+1], x_grid[j+2], y_grid[i], y_grid[i+1], y_grid[i+2], xy_elem)
            grad_test_fn_elem = grad_test_fn_linear2d(x_grid[j], x_grid[j+1], x_grid[j+2], y_grid[i], y_grid[i+1], y_grid[i+2], xy_elem)
            test_fn_values = test_fn_values.at[i, j].set(test_fn_elem)
            grad_test_fn_values = grad_test_fn_values.at[i, j].set(grad_test_fn_elem)
    
    quad_xy = quad_xy.reshape(N_nodes, N_quad, 2)
    quadw_xy = quadw_xy.reshape(N_nodes, N_quad)
    test_fn_values = test_fn_values.reshape(N_nodes, N_quad)
    grad_test_fn_values = grad_test_fn_values.reshape(N_nodes, N_quad, 2)
    

    a_fns = []
    for i in range(n_train):
        a_sample = inputs[:, :, i].T
        interpolator = RegularGridInterpolator(
            (x_interp_grid, y_interp_grid), 
            a_sample, 
            method='linear',
            fill_value=0.0
        )
        a_fns.append(interpolator)
    
    a_data = jnp.zeros((N_nodes, N_quad, n_train))
    for i in range(n_train):
        a_val_flat = a_fns[i](quad_xy.reshape(-1, 2))
        a_val = a_val_flat.reshape(N_nodes, N_quad)
        a_data = a_data.at[:, :, i].set(a_val)

    if jnp.any(jnp.isnan(a_data)):
        raise ValueError("NaN values found in a_data")
        
    f_data = jnp.ones((N_nodes, N_quad), dtype=jnp.float32)
    
    key = jax.random.PRNGKey(config.seed)

    print('\nTotal number of input functions', inputs.shape[-1])
    
    train_pinn_config = config.pinn.train
    
    quad_points   = {'residual': quad_xy, 'boundary': xy_bc}
    quad_weights = {'residual': quadw_xy, 'boundary': weights_bc}

    w_resid_pinn = train_pinn_config.w_residual
    w_bc_pinn = train_pinn_config.w_bc
    alpha = train_pinn_config.alpha
    weights_pinn = {'residual': w_resid_pinn, 'boundary': w_bc_pinn}
    
    train_pinn_steps = train_pinn_config.max_steps

    train_rebano_config = config.train

    grad_u_precomp = jnp.zeros((num_neurons, N_nodes, N_quad, 2))
    u_bc_precomp   = jnp.zeros((num_neurons, 4, N_bc))
    
    pinn_list = []
    
    max_losses   = []
    idx_list     = []
    
    try:
        use_pmap    = train_rebano_config.use_pmap
        batch_size  = train_rebano_config.batch_size
        max_devices = train_rebano_config.max_devices
    except AttributeError:
        
        use_pmap    = True
        batch_size  = 100
        max_devices = None
        
    n_devices = len(available_devices)
    if max_devices is not None:
        n_devices = min(n_devices, max_devices)
        available_devices = available_devices[:n_devices]  
          
    pinn_sample   = PINN(config.pinn)
    dummy_key     = jax.random.PRNGKey(0)
    sample_params = pinn_sample.init(dummy_key, quad_points['residual'])
    print("# of params in each PINN:", count_params(sample_params))
        
    print("\nStart ReBaNO training ...\n")
    print("Total number of samples:", inputs.shape[-1])
    print("Expected number of neurons:", num_neurons)
    
    print("******************************************************")
    
    if train_rebano_config.resume_training:
        num_pretrain_neurons = train_rebano_config.num_pretrain_neurons
        idx_list_stored = np.load(config.train.load_dir + "training_log_idx_list.npy")
        max_losses_stored = np.load(config.train.load_dir + "training_log_max_losses.npy")
        checkpoint_path = f"{config.train.load_dir}" + "pinn/"
        for i in range(num_pretrain_neurons):
            pinn_ckpt = load_checkpoint(f"{checkpoint_path}" + f"darcy_pinn_{i+1}")
            pinn_list.append(pinn_ckpt)
            pinn_config_loaded = ConfigDict(pinn_ckpt['metadata']['pinn_config'])
            pinn = PINN(pinn_config_loaded)
            apply_fn = get_u(pinn.apply)
            params   = pinn_ckpt['params']
            grad_u_vals = vmap(lambda x: compute_grad_u(apply_fn, params, x))(quad_points['residual']).squeeze(-2)
            grad_u_precomp = grad_u_precomp.at[i].set(grad_u_vals)
            u_bc_values = vmap(lambda x: apply_fn(params, x))(xy_bc_flatten).squeeze(-1)
            u_bc_values = u_bc_values.reshape(4, N_bc)
            u_bc_precomp = u_bc_precomp.at[i].set(u_bc_values)
            idx_list.append(int(idx_list_stored[i]))
            max_losses.append(max_losses_stored[i])
            
        print(f"{num_pretrain_neurons} pre-trained PINNs loaded from {checkpoint_path}")
    else:
        num_pretrain_neurons = 0
        
    idx_max = idx_list[-1] if idx_list else 0
    keys = jax.random.split(key, num_neurons)
    
    t0 = time.perf_counter()
    for i, key in enumerate(keys[num_pretrain_neurons:]):
        n = i + num_pretrain_neurons
        pinn   = PINN(config.pinn)
        a_pinn = a_data[..., idx_max]
        # a_pinn = jnp.ones((N_nodes, N_quad), dtype=jnp.float32)
        # f_data = 2 * jnp.pi**2 * jnp.sin(jnp.pi * quad_xy)[..., 0] * jnp.sin(jnp.pi * quad_xy[..., 1])
        print(f"\nStart training PINN #{n+1} with input function index {idx_max} ...\n")
        
        loss_data = {'residual': {'a': a_pinn, 'f': f_data}, 'boundary': None}
        
        
        pinn_state = create_train_state(key, pinn, train_pinn_config,
                                        quad_points['residual'], loss_data)
        
        t0_train_pinn = time.perf_counter()
        pinn_state = train_pinn(pinn_state, quad_points, quad_weights,
                                test_fn_values, grad_test_fn_values,
                                G, v1,
                                weights_pinn, alpha,
                                max_steps=train_pinn_steps, wandb_config=config.wandb)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(), pinn_state.params,
        )
        t1_train_pinn = time.perf_counter()

        print(f"\nPINN #{n+1} training complete!")
        print(f"PINN #{n+1} training time: {t1_train_pinn - t0_train_pinn:.4f} seconds\n")
        
        apply_fn = get_u(pinn_state.apply_fn)
        params   = pinn_state.params
        u_fn     = vmap(lambda x: apply_fn(params, x))
    
        u_pred   = u_fn(xy_test).squeeze()
        sol      = outputs[:, idx_max]
        # sol = jnp.sin(jnp.pi * xy_test[:, 0]) * jnp.sin(jnp.pi * xy_test[:, 1])
        l2_error = jnp.linalg.norm(u_pred - sol) / jnp.linalg.norm(sol)
        print(f"Relative L2 Error: {l2_error:.6e}")
    
        metadata = {
            'training resolution': a_pinn.shape[0]*a_pinn.shape[1],
            'input function': a_pinn,
            'l2_error': float(l2_error),
            'pinn_config': config.pinn.to_dict()
        }
    
        save_pinn_checkpoint(pinn_state, config.pinn.train.save_dir + f"darcy_pinn_{n+1}", metadata)
        
        pinn_ckpt = {
            'params': pinn_state.params,
            'metadata': metadata
        }
        pinn_list.append(pinn_ckpt)

        grad_u_values = vmap(lambda x: compute_grad_u(apply_fn, params, x))(quad_points['residual']).squeeze(-2)
        grad_u_precomp = grad_u_precomp.at[n].set(grad_u_values)
        u_bc_values = vmap(lambda x: apply_fn(params, x))(xy_bc_flatten).squeeze(-1)
        u_bc_values = u_bc_values.reshape(xy_bc.shape[0], N_bc)
        u_bc_precomp = u_bc_precomp.at[n].set(u_bc_values)
        
    
        print(f'\nStart finding input function #{n+2} for ReBaNO training...\n')
        
        t0_train_rebano = time.perf_counter()
        idx_max, loss_max = train_rebano(train_rebano_config, 
                                         pinn_list, quad_points, quad_weights, a_data, f_data, 
                                         test_fn_values, grad_test_fn_values,
                                         grad_u_precomp[:n+1], u_bc_precomp[:n+1], G, v1,
                                         available_devices, use_pmap, batch_size, wandb_config=config.wandb)
        t1_train_rebano = time.perf_counter()
        
        print(f"Search for the next input function time: {t1_train_rebano - t0_train_rebano:.4f} seconds\n")
        
        max_losses.append(float(loss_max))
        idx_list.append(idx_max)
        
        np.save(config.train.save_dir + "training_log_idx_list.npy", np.array(idx_list))
        np.save(config.train.save_dir + "training_log_max_losses.npy", np.array(max_losses))
    t1 = time.perf_counter()
    print('\nReBaNO Training Complete!')
    print(f"Total training time: {t1 - t0:.4f} seconds\n")
    idx_list = [0] + idx_list[:-1]
    for i in range(num_neurons):
        print(f"#{i+1} PINN with input function index {idx_list[i]} and max loss {max_losses[i]:.5e} (using {i+1} PINNs)")
    if config.wandb.enabled:
        try:
            wandb.log({
                'config': config.to_dict(),
                'idx_max_list': idx_list,
                'max_losses': max_losses
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error finishing wandb: {e}")
            pass
        

if __name__ == "__main__":
    main()

    
    


