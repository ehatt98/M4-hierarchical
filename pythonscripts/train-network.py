#### misc
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import time
from itertools import product

#### graphical
import matplotlib.pyplot as plt
import corner

#### ML
import sklearn
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras import layers

##### poke gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"

physical_devices = tf.config.list_physical_devices("GPU") 

tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpu0usage = tf.config.experimental.get_memory_info("GPU:0")["current"]

print("Current GPU usage:\n"
     + " - GPU0: " + str(gpu0usage) + "B\n")


df_full = pd.read_hdf('../grids/Chiara.hdf5', key='df') ## edit for your grid!!

#### define inputs
inputs = ['massini', 'zini', 'yini', 'alphaMLT', 'age', 'eta', 'alphaFe']

#### define outputs
classical_outputs = ['FeH', 'LPhot', 'Teff']
astero_outputs = ['numax', 'dnuSer'] # 10 modes for now

outputs = classical_outputs+astero_outputs

df = df_full[inputs+outputs]

df_norm = (df - df.min())/(df.max() - df.min())

## check df_norm.describe looks reasonable (min=0, max=1):
df_norm.describe()

#### train/test split with seed 
seed = 42

df_train = df_norm.sample(frac=0.95, random_state=seed)
df_test = df_norm.drop(df_train.index)

df_train_inputs, df_val_inputs, df_train_outputs, df_val_outputs = sklearn.model_selection.train_test_split(df_train[inputs],df_train[outputs], test_size = 0.05, random_state=seed)

print("Training set: ", len(df_train_inputs))
print("Validation set: ", len(df_val_inputs))
print("Test set: ", len(df_test))

######## define architecture:
model_name = 'simple_network'
n_dense_layers = 6 #number of dense layers
dense_layer_units = 64 #neurons per dense layer
Nepochs = 10000
learning_rate = 0.001
###### Checkpointing

checkpoint_dir = f'./checkpoints/chk-{model_name}-nlayers-{n_dense_layers}-nunits-{dense_layer_units}-epochs-{Nepochs}-lrate-{learning_rate}.model.keras'

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_dir, verbose = 1)

######## map out model architecture
#### input layer
nn_input = keras.Input(shape=(len(inputs),))

#### dense layer(s)
for n_dense_layer in range(n_dense_layers):
    if n_dense_layer == 0:
        dense_layer = layers.Dense(dense_layer_units, activation='relu')(nn_input)
    else:
        dense_layer = layers.Dense(dense_layer_units, activation='relu')(dense_layer)

#### output layer
nn_output =  layers.Dense(len(outputs), activation='linear')(dense_layer)

######## store architecture as keras model
model = keras.Model(inputs=nn_input, outputs=nn_output, name=model_name)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir = f'../logs/log-{model_name}-nlayers-{n_dense_layers}-nunits-{dense_layer_units}-epochs-{Nepochs}-lrate-{learning_rate}')

model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

history = model.fit(df_train_inputs,
          df_train_outputs,
          validation_data=(df_val_inputs,df_val_outputs),
          batch_size=4096, #change higher
          verbose=1,
          epochs=Nepochs,
          shuffle=True, callbacks = [tb_callback, cp_callback]) 
