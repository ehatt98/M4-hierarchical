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

from utils import norm, unnorm


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

    
    def model(self, x):

        return jax_emulator(self.weights, self.bias, x)

    
    def __call__(self, x):
        if self.normalised == True:
            return self.model(x)
        else:

            outs = []

            for j in range(len(x)):

                normed_x = jnp.array([norm(x[j][i], self.inputs[i], self.normconsts) for i in range(len(self.inputs))])

                normed_out = self.model(normed_x)

                out_= [unnorm(normed_out[i], self.outputs[i], self.normconsts) for i in range(len(self.outputs))]

                out_[1] = 10**out_[1]

                outs.append(jnp.array(out_))

            out = jnp.array(outs)

            return out


 
