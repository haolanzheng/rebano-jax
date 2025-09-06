import os
if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
    os.environ.setdefault("JAX_NUM_CPU_DEVICES", "5")
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap, devices, local_device_count
from ml_collections import ConfigDict
import sys
from functools import partial
from tqdm import tqdm
import wandb
import time

try:
    from ..models.nets import PINN, ReBaNO
    from .poisson_losses import *
    from .poisson_train import prepare_pmap_batch
    from ..train.training import make_step, create_train_state, make_step_rebano
    from ..train.precomp import compute_lap_u_scalar
    from ..configs.poisson_config import get_test_config
    from ..utils.grids import spatial_grid1d, spatial_grid1d_bc
    from ..utils.utilities import get_u, save_pinn_checkpoint, load_checkpoint
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.nets import PINN, ReBaNO
    from poisson_losses import *
    from poisson_train import prepare_pmap_batch
    from train.training import make_step, create_train_state, make_step_rebano
    from train.precomp import compute_lap_u_scalar
    from configs.poisson_config import get_test_config
    from utils.grids import spatial_grid1d, spatial_grid1d_bc
    from utils.utilities import get_u, save_pinn_checkpoint, load_checkpoint


def test_rebano(config_rebano, pinns, col_points, quad_weights,
                 f_data, u_xx_precomp, u_bc_precomp, available_devices,
                 use_pmap, batch_size, wandb_config=None):
    """Train the ReBaNO on the Poisson problem with pmap parallelization."""
    
    n_devices = len(available_devices)
    if not use_pmap or n_devices == 1:
        print("Using sequential processing")
        return test_rebano_sequential(config_rebano, pinns, 
                                      col_points, quad_weights,
                                       f_data, u_xx_precomp, u_bc_precomp, wandb_config)

    print(f"Using pmap with {n_devices} devices, batch size {batch_size}")
    return test_rebano_pmap(config_rebano, pinns, col_points,
                             quad_weights, f_data,
                             u_xx_precomp, u_bc_precomp, 
                             n_devices, batch_size, available_devices, wandb_config)


def test_rebano_pmap(config_rebano, pinns, col_points, quad_weights,
                     f_data, u_xx_precomp, u_bc_precomp,
                      n_devices, batch_size, available_devices, wandb_config):
    """Parallel ReBaNO training using pmap."""
    try: 
        max_steps = config_rebano.max_steps
    except AttributeError:
        max_steps = 5000
        
    dummy_key = jax.random.PRNGKey(0)
    alpha = config_rebano.alpha
    update_weights = config_rebano.update_weights
    
    n_samples = f_data.shape[1]
    n_neurons = len(pinns)
    
    def test_single_sample(f_sample):
        f_sample = f_sample.reshape(-1, 1)
        c_initial = jnp.full((n_neurons,), 1.0/n_neurons)
        rebano = ReBaNO(pinns, c_initial)
        
        loss_data = {'residual': {'f': f_sample}, 'boundary': None}
        loss_weights = {'residual': config_rebano.w_residual, 
                       'boundary': config_rebano.w_bc}
        
        state = create_train_state(dummy_key, rebano, config_rebano,
                                   col_points['residual'], loss_data)
        
        def boundary_loss_rebano(params, quad_weights,  loss_data=None):
            return poisson_boundary_loss_precomp(params, quad_weights, u_bc_precomp)

        def boundary_loss_grad_rebano(params, quad_weights, loss_data=None):
            return poisson_boundary_loss_grad(params, quad_weights, u_bc_precomp)

        def residual_loss_rebano(params, quad_weights, loss_data):
            return poisson_residual_loss_precomp(params, quad_weights, u_xx_precomp, loss_data)

        def residual_loss_grad_rebano(params, quad_weights, loss_data):
            return poisson_residual_loss_grad(params, quad_weights, u_xx_precomp, loss_data)

        loss_fns = {'residual': residual_loss_rebano, 'boundary': boundary_loss_rebano}
        grad_fns = {'residual': residual_loss_grad_rebano, 'boundary': boundary_loss_grad_rebano}
        
        step_fn = make_step_rebano(loss_fns, grad_fns)
        
        def loop_body(step, carry):
            state, loss_weights = carry
            state, loss_weights, metrics = step_fn(state, quad_weights, loss_weights, adaptive_weights=update_weights, alpha=alpha)
            return (state, loss_weights)
        
        # max_steps = 2000
        final_state, final_loss_weights = jax.lax.fori_loop(0, max_steps, loop_body, (state, loss_weights))
        
        # Get final loss
        final_state, _, final_metrics = step_fn(final_state, quad_weights, final_loss_weights, adaptive_weights=update_weights, alpha=alpha)
        final_loss = final_metrics['true loss']
        
        return final_state, final_loss
    
    def test_batch_samples(f_batch):
        batch_states, batch_losses = vmap(test_single_sample)(f_batch)
        return batch_states, batch_losses 
    
    pmap_test_batch = pmap(test_batch_samples, axis_name='device',
                            devices=available_devices)
    
    warm_f = jnp.zeros((n_devices, batch_size, f_data.shape[0]), dtype=f_data.dtype)
    warm_states, warm_losses = pmap_test_batch(warm_f)
    jax.block_until_ready(warm_losses)

    samples_per_device_batch = batch_size
    total_batch_size = samples_per_device_batch * n_devices
    n_batches = (n_samples + total_batch_size - 1) // total_batch_size

    processed_samples = 0
    
    print(f"Processing {n_samples} samples in {n_batches} parallel batches")
    print(f"Batch size per device: {samples_per_device_batch}, Total parallel: {total_batch_size}")
    
    states = []
    pbar_batches = tqdm(range(n_batches), desc="Parallel ReBaNO Training",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx in pbar_batches:
        start_idx = batch_idx * total_batch_size
        end_idx = min(start_idx + total_batch_size, n_samples)
        
        if start_idx >= n_samples:
            break
        
        batch_f_data = prepare_pmap_batch(
            f_data, start_idx, end_idx, samples_per_device_batch, n_devices
        )
        
        try:
            batch_states, batch_losses = pmap_test_batch(batch_f_data)
            jax.block_until_ready(batch_states)
            
            actual_samples_in_batch = min(end_idx - start_idx, total_batch_size)
            
            flat_states = jax.tree_util.tree_map(
                lambda a: a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:]),
                batch_states
            )
            flat_states = jax.tree_util.tree_map(lambda a: a[:actual_samples_in_batch], flat_states)

            m = jax.tree_util.tree_leaves(flat_states)[0].shape[0]
            states.extend([jax.tree_util.tree_map(lambda a, i=i: a[i], flat_states) for i in range(m)])
            
            flat_losses = batch_losses.flatten()

            valid_mask = jnp.isfinite(flat_losses)
            valid_losses = flat_losses[valid_mask]
            
            if len(valid_losses) > 0:
                
                processed_samples += actual_samples_in_batch
                
                pbar_batches.set_postfix({
                    'Processed': f'{processed_samples}/{n_samples}'
                })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    pbar_batches.close()
    
    print(f'\nParallel ReBaNO Fine Tuning Complete!')
    print(f'Processed {processed_samples}/{n_samples} samples')
    
    if wandb_config and wandb_config.enabled:
        log_data = {
            'processed_samples': processed_samples,
            'total_samples': n_samples
        }
        wandb.log(log_data)
    
    return states


def test_rebano_sequential(config_rebano, pinns, col_points, 
                           quad_weights, f_data, u_xx_precomp, u_bc_precomp, wandb_config):
    """Sequential ReBaNO training - fallback when pmap is not available."""
    try:
        max_steps = config_rebano.max_steps
        tol_grad  = config_rebano.tol_grad
    except AttributeError:
        max_steps = 5000
        tol_grad  = 1e-9
    
    def boundary_loss_rebano(params, quad_weights, loss_data=None):
        return poisson_boundary_loss_precomp(params, quad_weights, u_bc_precomp)

    def boundary_loss_grad_rebano(params, quad_weights, loss_data=None):
        return poisson_boundary_loss_grad(params, quad_weights, u_bc_precomp)

    def residual_loss_rebano(params, quad_weights, loss_data):
        return poisson_residual_loss_precomp(params, quad_weights, u_xx_precomp, loss_data)

    def residual_loss_grad_rebano(params, quad_weights, loss_data):
        return poisson_residual_loss_grad(params, quad_weights, u_xx_precomp, loss_data)

    n_samples = f_data.shape[1]
    n_neurons = len(pinns)
    
    dummy_key = jax.random.PRNGKey(0)
    update_weights = config_rebano.update_weights
    alpha = config_rebano.alpha

    pbar_samples = tqdm(range(n_samples), desc="Sequential ReBaNO Training",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    loss_fns = {'residual': residual_loss_rebano, 'boundary': boundary_loss_rebano}
    grad_fns = {'residual': residual_loss_grad_rebano, 'boundary': boundary_loss_grad_rebano}
    
    states = []
    
    for sample_idx in pbar_samples:
        f_sample = f_data[:, sample_idx:sample_idx+1]
        c_initial = jnp.full((n_neurons,), 1.0/n_neurons)
        rebano = ReBaNO(pinns, c_initial)
        
        loss_data = {'residual': {'f': f_sample}, 'boundary': None}
        loss_weights = {'residual': config_rebano.w_residual, 'boundary': config_rebano.w_bc}
        
        state = create_train_state(dummy_key, rebano, config_rebano,
                                   col_points['residual'], loss_data)

        step_fn = make_step_rebano(loss_fns, grad_fns)
        
        for step in range(max_steps):
            state, loss_weights, metrics = step_fn(state, quad_weights, loss_weights, adaptive_weights=update_weights, alpha=alpha)
            jax.block_until_ready(metrics['true loss'])
            
            loss_value = metrics['true loss']
            
            if metrics['grad_norm'] < tol_grad:
                break
        
        states.append(state)
        pbar_samples.set_postfix({
            'total samples': n_samples,
        })
    
    pbar_samples.close()
    
    print(f'\nSequential ReBaNO Testing Complete!')
    
    if wandb_config and wandb_config.enabled:
        log_data = {'total samples': n_samples}
        wandb.log(log_data)

    return states

def pred_and_error(states, x_eval, u_exact):
    apply_fn = states[0].apply_fn
    fn = get_u(apply_fn)

    @jit
    def predict(params, x_eval):
        return jax.vmap(lambda x: fn(params, x))(x_eval).reshape(-1)  # (Nx,)

    preds_list = []
    rel_list   = []

    for i, st in enumerate(states):
        pred = predict(st.params, x_eval)  
        rel  = jnp.linalg.norm(pred - u_exact[:, i]) / jnp.linalg.norm(u_exact[:, i])
        preds_list.append(pred)      # each is shape (Nx,)
        rel_list.append(rel)         # each is scalar

    preds      = jnp.stack(preds_list, axis=1)
    rel_errors = jnp.stack(rel_list, axis=0)

    return preds, rel_errors

def main():
    config = get_test_config()

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
                name=f"poisson_rebano_neurons_{config.num_neurons}_test",
                config=config.to_dict()
            )
            print(f"Wandb initialized successfully for project: {config.wandb.project}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing testing without wandb logging...")
            # Disable wandb for this run
            config.wandb.enabled = False

    num_neurons = config.num_neurons

    # loading data
    offset  = config.data.offset
    n_test  = config.data.n_samples

    inputs  = np.load(config.data.inputs_dir)[::config.data.sub_x, offset:n_test+offset].astype(np.float32)
    outputs = np.load(config.data.outputs_dir)[::config.data.sub_x, offset:n_test+offset].astype(np.float32)  

    inputs  = jnp.array(inputs.astype(np.float32))
    outputs = jnp.array(outputs.astype(np.float32))

    Nx = inputs.shape[0]
    Xi, Xf = config.domain.Xi, config.domain.Xf
    x_resid, quadw_resid = spatial_grid1d(Xi, Xf, Nx)
    x_bc, quadw_bc       = spatial_grid1d_bc(Xi, Xf)

    print('\ntotal number of input functions', inputs.shape[-1])
    
    col_points = {'residual': x_resid, 'boundary': x_bc}
    quad_weights = {'residual': quadw_resid, 'boundary': quadw_bc}
    
    test_rebano_config = config.test
    
    u_xx_precomp = jnp.zeros((num_neurons, x_resid.shape[0], 1))
    u_bc_precomp = jnp.zeros((num_neurons, x_bc.shape[0], 1))
    
    pinn_list = []
    
    try:
        use_pmap    = test_rebano_config.use_pmap
        batch_size  = test_rebano_config.batch_size
        max_devices = test_rebano_config.max_devices
    except AttributeError:
        use_pmap    = True
        batch_size  = 100
        max_devices = None
        
    n_devices = len(available_devices)
    if max_devices is not None:
        n_devices = min(n_devices, max_devices)
        available_devices = available_devices[:n_devices]  
        
    print("\nStart ReBaNO testing ...\n")
    print("Total number of test samples:", inputs.shape[1])
    
    print("******************************************************")
    
    checkpoint_path = f"{test_rebano_config.load_pinn_dir}"
    for i in range(num_neurons):
        pinn_ckpt = load_checkpoint(f"{checkpoint_path}" + f"poisson_pinn_{i+1}")
        pinn_list.append(pinn_ckpt)
        pinn_config_loaded = ConfigDict(pinn_ckpt['metadata']['pinn_config'])
        pinn = PINN(pinn_config_loaded)
        apply_fn = get_u(pinn.apply)
        params   = pinn_ckpt['params']
        u_xx = compute_lap_u_scalar(apply_fn, params, x_resid).reshape(-1, 1)
        u_xx_precomp = u_xx_precomp.at[i].set(u_xx)
        u_bc = vmap(lambda x: apply_fn(params, x))(x_bc).reshape(-1, 1)
        u_bc_precomp = u_bc_precomp.at[i].set(u_bc)
        
    print(f"Number of neurons loaded: {num_neurons}\n")
  
    t0 = time.perf_counter()
    states = test_rebano(test_rebano_config, pinn_list, 
                        col_points, quad_weights, inputs, u_xx_precomp, u_bc_precomp, available_devices, use_pmap, batch_size, wandb_config=config.wandb)
    t1 = time.perf_counter()
    
    print('\nReBaNO testing complete!')
    print(f"Total test time: {t1 - t0:.4f} seconds\n")
    
    predictions, rel_errors = pred_and_error(states, x_resid, outputs)
    predictions, rel_errors = jax.device_get(predictions), jax.device_get(rel_errors)

    os.makedirs(os.path.dirname(test_rebano_config.save_dir) if os.path.dirname(test_rebano_config.save_dir) else '.', exist_ok=True)

    np.save(test_rebano_config.save_dir + f"poisson_rebano_predictions_s{Nx}.npy", predictions)
    np.save(test_rebano_config.save_dir + f"poisson_rebano_rel_errors_s{Nx}.npy", rel_errors)
    
    print("ReBaNO predictions saved:", test_rebano_config.save_dir + f"poisson_rebano_predictions_s{Nx}.npy")
    print("ReBaNO relative errors saved:", test_rebano_config.save_dir + f"poisson_rebano_rel_errors_s{Nx}.npy")
    
    mean_rel_err = np.mean(rel_errors)
    max_rel_err  = np.max(rel_errors)
    
    print(f"Mean relative error over {inputs.shape[1]} samples: {mean_rel_err:.5e}")
    print(f"Largest relative error over {inputs.shape[1]} samples: {max_rel_err:.5e}")

    if config.wandb.enabled:
        try:
            wandb.log({
                'config': config.to_dict(),
                'mean_rel_err': mean_rel_err,
                'max_rel_err': max_rel_err
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error finishing wandb: {e}")
            pass
        

if __name__ == "__main__":
    main()

    
    


