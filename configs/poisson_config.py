import numpy as np 
import jax 
import jax.numpy as jnp 

import flax.linen as nn
from jax.nn.initializers import glorot_normal

import ml_collections as mlc
from ml_collections import ConfigDict


def get_train_config():
    """ Fetch the training configurations """
    
    config = ConfigDict()
    config.seed        = 1234
    config.num_neurons = 8
    
    config.domain = domain = ConfigDict()
    domain.Xi = 0.0
    domain.Xf = 1.0
    domain.sampling_method = 'uniform'
    
    config.train = train = ConfigDict()
    train.resume_training      = False
    train.num_pretrain_neurons = 0
    train.load_dir             = 'ckpts/'
    train.use_pmap             = True
    train.batch_size           = 100
    train.max_devices          = 8

    train.lr             = 0.001
    train.optimizer      = 'Adam'
    train.max_steps      = 2000
    train.decay_steps    = 1000
    train.decay_rate     = 0.5
    train.update_weights = False
    train.w_residual     = 1.0
    train.w_bc           = 1.0
    train.alpha          = 1.0
    train.save_dir       = 'ckpts/'
    
    config.wandb   = wandb = ConfigDict()
    wandb.project  = 'poisson_rebano'
    wandb.entity   = 'haolanzheng-university-of-massachusetts-dartmouth'
    wandb.enabled  = True
    wandb.log_freq = 1000
    
    # data loading
    config.data      = data = ConfigDict()
    data.inputs_dir  = '../data/poisson/poisson_input_f_K10_s128.npy'
    data.outputs_dir = '../data/poisson/poisson_output_u_K10_s128.npy'
    data.sub_x       = 1
    data.sub_y       = 1
    data.offset      = 0
    data.n_samples   = 1000
    
    # pinn configs
    config.pinn       = pinn = ConfigDict()
    pinn.arch        = 'PirateNet'
    pinn.num_layers  = 1
    pinn.hidden_dim  = 32
    pinn.out_dim     = 1
    pinn.embed_scale = 0.1
    pinn.embed_dim   = 64
    pinn.init_fn     = 'glorot_normal'
    pinn.activation  = 'tanh'
    pinn.fact_weight = True

    # pinn training configs
    pinn.train             = pinn_train = ConfigDict()
    pinn_train.optimizer   = 'Adam'
    pinn_train.lr          = 0.001
    pinn_train.decay_steps = 5000
    pinn_train.decay_rate  = 0.5
    pinn_train.max_steps   = 10000
    pinn_train.save_dir    = 'ckpts/pinn/'
    pinn_train.update_weights = True
    pinn_train.w_residual     = 10.0
    pinn_train.w_bc           = 1.0
    pinn_train.alpha          = 0.8

    return config
    
    
def get_test_config():
    """ Fetch the test configurations """
    
    config = ConfigDict()
    config.num_neurons = 8
    
    config.domain = domain = ConfigDict()
    domain.Xi = 0.0
    domain.Xf = 1.0
    domain.sampling_method = 'uniform'
     
    config.test = test = ConfigDict()
    test.load_pinn_dir = 'ckpts/pinn/'
    test.use_pmap      = True
    test.batch_size    = 100
    test.max_devices   = 8
    
    test.lr             = 0.001
    test.optimizer      = 'Adam'
    test.max_steps      = 5000
    test.decay_steps    = 5000
    test.decay_rate     = 0.5
    test.update_weights = False
    test.w_residual     = 1.0
    test.w_bc           = 1.0
    test.alpha          = 1.0
    test.save_dir       = '../data/poisson/'
    
    config.wandb   = wandb = ConfigDict()
    wandb.project  = 'poisson_rebano'
    wandb.entity   = 'haolanzheng-university-of-massachusetts-dartmouth'
    wandb.enabled  = True
    wandb.log_freq = 1000
    
    # data loading
    config.data      = data = ConfigDict()
    data.inputs_dir  = '../data/poisson/ood/poisson_input_f_K10_s128.npy'
    data.outputs_dir = '../data/poisson/ood/poisson_output_u_K10_s128.npy'
    data.sub_x       = 1
    data.sub_y       = 1
    data.offset      = 0
    data.n_samples   = 1000

    return config    