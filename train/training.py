import jax
import jax.numpy as jnp
from jax import vmap, pmap, value_and_grad, jit

import optax
import flax
from flax.training import train_state
from ml_collections import ConfigDict

from typing import Any, Callable, Dict, Tuple, Union
from flax.core.frozen_dict import FrozenDict

from functools import partial

Array  = jax.Array
PyTree = Union[FrozenDict, Dict]
Batch  = Dict[str, jax.Array]


class TrainStateWithData(train_state.TrainState):
    loss_data: Dict[str, Any] = None


def create_optimizer(config: ConfigDict):
    """Create an optimizer object from optax."""
    if config.optimizer == 'Adam':
        lr = optax.exponential_decay(init_value=config.lr,
                                     transition_steps=config.decay_steps,
                                     decay_rate=config.decay_rate)
        optimizer = optax.adam(learning_rate=lr)
        optimizer = optax.with_extra_args_support(optimizer)

    elif config.optimizer == 'LBFGS':
        optimizer = optax.lbfgs()
        optimizer = optax.with_extra_args_support(optimizer)
        
    else:
        raise NotImplementedError(f"Provided optimizer {config.optimizer} is not supported!")
    
    return optimizer

def create_train_state(key: jax.random.PRNGKey,
                   model: Any,
                   config: ConfigDict,
                   batch: Any,
                   loss_data: Dict[str, Any] = None) -> train_state.TrainState:
    """Create and initialize the model train state. Attach optional loss_data."""
    tx       = create_optimizer(config)
    params   = model.init(key, batch)
    apply_fn = model.apply
    state    = TrainStateWithData.create(apply_fn=apply_fn,
                                         params=params,
                                         tx=tx,
                                         loss_data=loss_data)
    
    return state
    

def compute_grads(apply_fn: Callable,
                      params: PyTree,
                      names: Tuple[str, ...],
                      fns: Tuple[Callable, ...],
                      batch_data: Batch,
                      quad_weights: Batch,
                      loss_data: Dict[str, Any]):
    """Compute only gradients without computing loss values using loss_data from state."""
    grads = {}
    for name, loss_fn in zip(names, fns):
        loss_wrapper = partial(loss_fn, apply_fn=apply_fn, batch_data=batch_data[name], quad_weights=quad_weights[name], loss_data=loss_data[name])
        grad = jax.grad(loss_wrapper)(params)
        grads[name] = grad
    return grads


def compute_loss_and_grads(apply_fn: Callable,
                         params: PyTree,
                         names: Tuple[str, ...],
                         fns: Tuple[Callable, ...],
                         batch_data: Batch,
                         quad_weights: Batch,
                         loss_data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, PyTree]]:
    """Compute individual losses and their gradients using loss_data from state."""
    losses = {}
    grads = {}
    for name, loss_fn in zip(names, fns):
        loss_wrapper = partial(loss_fn, apply_fn, batch_data=batch_data[name], quad_weights=quad_weights[name], loss_data=loss_data[name])
        loss_val, grad = value_and_grad(loss_wrapper)(params)
        losses[name] = loss_val
        grads[name] = grad
    return losses, grads

def update_weights(grads: Dict[str, PyTree],
                  prev_weights: Dict[str, float],
                  alpha: float=0.8) -> Dict[str, float]:
    """Update loss weights using gradient norm ratios.
    """
    grad_norms = {name: optax.global_norm(grad) for name, grad in grads.items()}
    total_norm = sum(grad_norms.values())
    
    weights = {
        name: (1.0 - alpha) * total_norm / (grad_norm + 1e-8) + alpha * prev_weights[name]
        for name, grad_norm in grad_norms.items()
    }
    
    return weights

def make_step(loss_fns: Dict[str, Callable]):
    """Create a training step function with adaptive loss weighting (reads state.loss_data)."""
    
    names = tuple(sorted(loss_fns.keys()))
    fns   = tuple(loss_fns[n] for n in names)

    @partial(jit, static_argnames=['adaptive_weights'])
    def step(state: TrainStateWithData,
                loss_weights: Dict[str, float],
                batch: Batch,
                quad_weights: Batch,
                adaptive_weights: bool=True,
                alpha: float=0.8) -> Tuple[TrainStateWithData, Dict[str, float], Dict[str, float]]:
        """Perform a single training step with adaptive loss weighting using state.loss_data."""
        losses, indiv_grads = compute_loss_and_grads(state.apply_fn, state.params, names, fns, batch, quad_weights, state.loss_data)
        if adaptive_weights:
            new_weights = update_weights(indiv_grads, loss_weights, alpha)
        else:
            new_weights = loss_weights
        total_loss = sum(new_weights[name] * losses[name] for name in names)

        total_grads = jax.tree_util.tree_map(
            lambda *grads: sum(new_weights[name] * grad for name, grad in zip(names, grads)),
            *[indiv_grads[name] for name in names]
        )
        
        grad_norm = optax.global_norm(total_grads)
        
        def total_loss_fn(p):                               
            return sum(
                new_weights[name] * fns[i](state.apply_fn, p, batch[name], state.loss_data[name])
                for i, name in enumerate(names)
            )
        
        updates, new_opt_state = state.tx.update(total_grads, 
                                                state.opt_state, 
                                                state.params,
                                                value=total_loss,
                                                grad=total_grads,
                                                value_fn=total_loss_fn)
        new_params = optax.apply_updates(state.params, updates)
        
        new_state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state
        )
        
        metrics = {
            'total_loss': total_loss,
            'grad_norm': grad_norm,
            **{f'{name}_loss': val for name, val in losses.items()},
            **{f'{name}_weight': val for name, val in new_weights.items()}
        }
        
        return new_state, new_weights, metrics
    
    return step


def make_step_rebano(loss_precomp: Dict[str, Callable], loss_grads_precomp: Dict[str, Callable]):
    """Create a training step function for ReBaNO using loss_data in state."""
    
    names = tuple(sorted(loss_precomp.keys()))
    loss_fns = tuple(loss_precomp[n] for n in names)
    grad_fns = tuple(loss_grads_precomp[n] for n in names)

    @partial(jit, static_argnames=['adaptive_weights'])
    def step(state: TrainStateWithData,
                quad_weights: Batch,
                loss_weights: Dict[str, float],
                adaptive_weights: bool=False,
                alpha: float=0.8) -> Tuple[TrainStateWithData, Dict[str, float], Dict[str, float]]:
        """Perform a single training step with loss_data in state."""
        losses = {name: loss_fns[i](state.params, quad_weights[name], state.loss_data[name]) for i, name in enumerate(names)}
        indiv_grads  = {name: grad_fns[i](state.params, quad_weights[name], state.loss_data[name]) for i, name in enumerate(names)}

        if adaptive_weights:
            new_weights = update_weights(indiv_grads, loss_weights, alpha)
        else:
            new_weights = loss_weights
            
        total_loss = sum(new_weights[name] * losses[name] for name in names)

        total_grads = jax.tree_util.tree_map(
            lambda *grads: sum(new_weights[name] * grad for name, grad in zip(names, grads)),
            *[indiv_grads[name] for name in names]
        )
        
        grad_norm = optax.global_norm(total_grads)
        true_loss = sum(losses[name] for name in names)
        
        def total_loss_fn(p):                               
            return sum(
                new_weights[name] * loss_fns[i](p, state.loss_data[name])
                for i, name in enumerate(names)
            )
        
        updates, new_opt_state = state.tx.update(total_grads, 
                                                 state.opt_state, 
                                                 state.params,
                                                 value=total_loss,
                                                 grad=total_grads,
                                                 value_fn=total_loss_fn)
        new_params = optax.apply_updates(state.params, updates)
        
        
        new_state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state
        )
        
        metrics = {
            'total_loss': total_loss,
            'true loss': true_loss,
            'grad_norm': grad_norm,
            **{f'{name}_loss': val for name, val in losses.items()},
            **{f'{name}_weight': val for name, val in new_weights.items()}
        }
        
        return new_state, new_weights, metrics
    
    return step