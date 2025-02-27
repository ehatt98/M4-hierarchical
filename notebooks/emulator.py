import numpy as np
import os
from pathlib import Path

import tensorflow as tf
import keras
from keras import layers

from jax import vmap, Array
from jax.typing import ArrayLike
from jax.nn import relu
import jax.numpy as jnp

from utils import norm, unnorm, norm_matrix, unnorm_matrix


def jax_emulator(weights, bias, x):
    x = jnp.array(x)
    for w, b in zip(weights[:-1], bias[:-1]):
        x = relu(jnp.matmul(x, w) + b)
    x = jnp.matmul(x, weights[-1]) + bias[-1]
    return x

def power10(x):

    return jnp.power(10, x)

class Emulator:

    def __init__(self, emulator_kwargs = {'model_name':None, 'n_dense_layers':None, 'dense_layer_units':None, 'Nepochs':None, 'lrate':None, \
        'loss':None, 'model_dir':None, 'checkpoint_dir':None, 'inputs': ['massini', 'zini', 'yini', 'alphaMLT', 'age', 'eta', 'alphaFe'], 'outputs':['FeH', 'logLPhot', 'Teff', 'numax', 'dnuSer'],\
        'normconsts':None}, normalised = True):
 
        for key, value in emulator_kwargs.items():
            setattr(self, key, value)

        self.normalised = normalised

        if self.checkpoint_dir is None:
            self.checkpoint_dir = f'{self.model_dir}/checkpoints/chk-{self.model_name}-nlayers-{self.n_dense_layers}-nunits-{self.dense_layer_units}-epochs-{self.Nepochs}-lrate-{self.lrate}-lossfunc-{self.loss}.model.keras'

        self.keras_model = tf.keras.models.load_model(self.checkpoint_dir)

        w_ = []

        b_ = []

        for i in range(1, self.n_dense_layers + 2):
            w_.append(self.keras_model.layers[i].get_weights()[0])
            b_.append(self.keras_model.layers[i].get_weights()[1])

        self.weights = [jnp.array(w) for w in w_]

        self.bias = [jnp.array(b) for b in b_]


        norm_mins_in = jnp.array([self.normconsts[i][0] for i in self.inputs])

        norm_maxes_in = jnp.array([self.normconsts[i][1] for i in self.inputs])

        self.norm_stack_in = jnp.vstack([norm_mins_in, norm_maxes_in])

        norm_mins_out = jnp.array([self.normconsts[i][0] for i in self.outputs])

        norm_maxes_out = jnp.array([self.normconsts[i][1] for i in self.outputs])

        self.norm_stack_out = jnp.vstack([norm_mins_out, norm_maxes_out])
        
        obs_values = np.full((len(self.outputs) + 1), None)
        
        obs_dict = {}
        
        for key, value in zip(self.outputs + ['LPhot'], obs_values):
            if key not in obs_dict:
                obs_dict[key] = value
            else:
                obs_dict[key].append(value)
        
        self.obs_dict = obs_dict

    
    def model(self, x):

        return jax_emulator(self.weights, self.bias, x)


    def __call__(self, x):
            if self.normalised == True:
                return self.model(x)

            else:

                normed_x = norm_matrix(x, self.norm_stack_in)

                normed_out = self.model(normed_x)

                out = unnorm_matrix(normed_out, self.norm_stack_out)

                return out






