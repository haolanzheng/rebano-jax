import numpy as np 
import jax 
import jax.numpy as jnp

from typing import Any, Callable

import pickle, os

from models.nets import PINN, ReBaNO

Array = jax.Array

def get_u(apply_fn: Callable):
    """ 
    Returns u_eval(params, xy) -> scalar, vector, or tensor.
    
    Input coordinates should have shape (N_points, n_dim), and when vmapped,
    each call to u_eval receives a single point with shape (n_dim,).
    """
    def u_eval(params: Any, xy: Array):

        xy_input = xy[None, :]
        
        output = apply_fn(params, xy_input)
        
        if output.ndim == 1:
            return output
        else:
            return output[0]
    
    return u_eval

def count_params(params):
    def count_single_param(param):
        return param.size
    
    param_counts = jax.tree_util.tree_map(count_single_param, params)
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y, param_counts, 0)
    
    return total_params

def print_model_summary(model, params, sample_input=None):
    """
    Print a summary of the model including parameter count and structure.
    
    Args:
        model: The model instance (PINN, ReBaNO, etc.)
        params: Model parameters from model.init()
        sample_input: Optional sample input to show input/output shapes
    """
    total_params = count_params(params)
    
    print("="*50)
    print(f"Model Summary: {type(model).__name__}")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    
    # Print parameter structure
    print("\nParameter Structure:")
    def print_param_info(path, param):
        path_str = '.'.join(str(k.key) for k in path)
        print(f"  {path_str}: {param.shape} ({param.size:,} params)")
    
    jax.tree_util.tree_map_with_path(print_param_info, params)
    
    # If sample input provided, show input/output shapes
    if sample_input is not None:
        try:
            output = model.apply(params, sample_input)
            print(f"\nInput shape: {sample_input.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"\nCould not determine I/O shapes: {e}")
    
    print("="*50)

def load_checkpoint(filepath: str):
    """Load model checkpoint."""
    
    with open(f"{filepath}.pkl", 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded: {filepath}.pkl")
    return checkpoint

def save_pinn_checkpoint(state, filepath: str, metadata: dict = None):
    """Save trained PINN checkpoint."""
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'params': state.params,
        'metadata': metadata or {}
    }
    
    with open(f"{filepath}.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {filepath}.pkl")

def load_pinn(checkpoint_path: str, model_config=None):
    """Load pinns as pure callable function.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration to recreate the PINN (if not in metadata)
    """
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['params']
    
    if 'metadata' in checkpoint and 'model_config' in checkpoint['metadata']:
        config = checkpoint['metadata']['model_config']
    elif model_config is not None:
        config = model_config
    else:
        raise ValueError("Model config not found in checkpoint metadata and not provided as argument")
    
    pinn = PINN(config)
    
    def model(x):
        """Model function: x -> u(x)"""
        u_fn = get_u(pinn.apply)
        return u_fn(params, x)
    
    return model

def save_rebano_checkpoint(rebano_state, rebano_instance, filepath: str, metadata: dict = None):
    """Save trained ReBaNO model checkpoint with coefficients and PINN data."""
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'coefficients': rebano_state.params['coefficients'],
        'ckpt_data': rebano_instance.ckpt_data,
        'c_initial': rebano_instance.c_initial,
        'metadata': metadata or {}
    }
    
    with open(f"{filepath}.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"ReBaNO checkpoint saved: {filepath}.pkl")

def load_rebano(checkpoint_path: str):
    """
    Load ReBaNO model as callable function.
    """
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    coefficients = checkpoint['coefficients']
    ckpt_data = checkpoint['ckpt_data']
    c_initial = checkpoint['c_initial']
    
    rebano = ReBaNO(ckpt_data=ckpt_data, c_initial=c_initial)
    
    params = {'coefficients': coefficients}
    
    def model(x):
        """ReBaNO model function: x -> Σᵢ cᵢ * u_pinn^i(x)"""
        u_fn = get_u(rebano.apply)
        return u_fn(params, x)
    
    return model

def prepare_pmap_batch(f_data, start_idx, end_idx, batch_size, n_devices, batch_axis=0):
    """Prepare data for pmap - reshape to (n_devices, batch_size, ...)."""
    actual_samples = end_idx - start_idx
    total_batch_size = batch_size * n_devices
    
    if batch_axis == -1:
        f_data = jnp.moveaxis(f_data, -1, 0)
    elif batch_axis == 0:
        f_data = f_data
    else:
        raise ValueError("batch_axis must be 0 or -1")
    
    if actual_samples < total_batch_size:
        f_padded = jnp.zeros((total_batch_size, *f_data.shape[1:]))
        f_padded = f_padded.at[:actual_samples, ...].set(f_data[start_idx:end_idx, ...])
        if actual_samples > 0:
            last_sample = f_data[end_idx-1:end_idx, ...]
            for i in range(actual_samples, total_batch_size):
                f_padded = f_padded.at[i:i+1, ...].set(last_sample)
        f_batch = f_padded
    else:
        f_batch = f_data[start_idx:start_idx + total_batch_size, ...]


    f_batch = f_batch.reshape(n_devices, batch_size, *f_data.shape[1:])

    return f_batch