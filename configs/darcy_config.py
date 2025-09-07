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
    config.num_neurons = 20
    
    config.domain = domain = ConfigDict()
    domain.Xi = 0.0
    domain.Xf = 1.0
    domain.Yi = 0.0
    domain.Yf = 1.0
    domain.Nelem_x = 9
    domain.Nelem_y = 9
    domain.N_quad  = 16
    domain.N_bc    = 100
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
    train.save_dir       = 'ckpts/'
    
    config.wandb   = wandb = ConfigDict()
    wandb.project  = 'darcy_rebano'
    wandb.entity   = 'haolanzheng-university-of-massachusetts-dartmouth'
    wandb.enabled  = False
    wandb.log_freq = 1000
    
    # data loading
    config.data      = data = ConfigDict()
    data.inputs_dir  = '../data/darcy/darcy_input_a_K8_s100.npy'
    data.outputs_dir = '../data/darcy/darcy_output_u_K8_s100.npy'
    data.sub_x       = 1
    data.sub_y       = 1
    data.offset      = 0
    data.n_samples   = 100
    
    # pinn configs
    config.pinn       = pinn = ConfigDict()
    pinn.arch        = 'PirateNet'
    pinn.num_layers  = 6
    pinn.hidden_dim  = 64
    pinn.out_dim     = 1
    pinn.embed_scale = 0.1
    pinn.embed_dim   = 256
    pinn.init_fn     = 'glorot_normal'
    pinn.activation  = 'sin'
    pinn.fact_weight = True

    # pinn training configs
    pinn.train             = pinn_train = ConfigDict()
    pinn_train.optimizer   = 'Adam'
    pinn_train.lr          = 0.005
    pinn_train.decay_steps = 5000
    pinn_train.decay_rate  = 0.8
    pinn_train.max_steps   = 40000
    pinn_train.save_dir    = 'ckpts/pinn/'
    pinn_train.update_weights = False
    pinn_train.w_residual     = 50.0
    pinn_train.w_bc           = 2.0
    pinn_train.alpha          = 0.9

    return config
    
    
def get_test_config():
    """ Fetch the test configurations """
    
    config = ConfigDict()
    config.num_neurons = 8
    
    config.domain = domain = ConfigDict()
    domain.Xi = 0.0
    domain.Xf = 1.0
    domain.Yi = 0.0
    domain.Yf = 1.0
    domain.Nelem_x = 8
    domain.Nelem_y = 8
    domain.N_quad  = 4
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
    test.save_dir       = '../data/darcy/'
    
    config.wandb   = wandb = ConfigDict()
    wandb.project  = 'darcy_rebano'
    wandb.entity   = 'haolanzheng-university-of-massachusetts-dartmouth'
    wandb.enabled  = True
    wandb.log_freq = 1000
    
    # data loading
    config.data      = data = ConfigDict()
    data.inputs_dir  = '../data/darcy/darcy_input_a_K8_s100.npy'
    data.outputs_dir = '../data/darcy/darcy_output_u_K8_s100.npy'
    data.sub_x       = 1
    data.sub_y       = 1
    data.offset      = 1000
    data.n_samples   = 1000

    return config    