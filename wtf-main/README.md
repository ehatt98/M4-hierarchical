# WTF - Without TensorFlow
a couple useful tools for moving away from TensorFlow dependency after training a model:

- `tf_to_dict` goes from a saved keras model to a streamlined dict containing important architecture information, weights, bases, and layer order

- `compile_from_dict/numpy_compile` - compile a network with NumPy backend, perform forward pass with `forward_pass`

- `compile_from_dict/jax_compile` - compile a network with JAX backend, perform forward pass with `forward_pass` or JIT-compile and pass with `jit_forward_pass`

See the notebooks `tf_to_dict.ipynb` and `compile_from_dict.ipynb` for examples of usage on the example model in `models`.

## recommended usage
1) in training env (with TensorFlow installed): immediately after training and saving a TensorFlow model, `tf_model`, convert to dict with `wtf_dict = tf_to_dict(tf_model)` and save to .json or pickle

2) in inference env (streamlined env without TensorFlow): load in saved json/pickle dict from `tf_to_dict` and compile with either `wtf_model = numpy_compile(wtf_dict)` or `wtf_model = jax_compile(wtf_dict)` (I strongly advise compiling once in NumPy and assuring expected behaviour before moving to JAX (see gotchas))

3) in inference env: perform forward pass for array of inputs `x` with compiled model with `wtf_model.forward_pass(x)`

4) If compiled in JAX (i.e. `wtf_model = jax_compile(wtf_dict)`), then you have the option to JIT compile the forward pass by calling `wtf_model.jit_forward_pass(x)` instead

## gotchas
- !This code was hastily written! Please check the model interpretation printouts from `tf_to_dict` match your expected network structure before trying `compile_from_dict` steps
- The NumPy version of `forward_pass` contains an assert to check that the batch shape passed to the network which is not present in the JAX version (to allow compilation) - therefore, I strongly advice checking behaviour of `forward_pass` in NumPy before switching to JAX!
- You will need to hard code your data scaling and any custom objects (like a final PCA reprojection layer, for example) as a wrapper around your `wtf_model.forward_pass(x)` to get expected results
- Currently only supports simple sequential or branching networks
- Currently only implemented `elu` activation
- Make sure `x` is defined in appropriate backend (i.e. `np.array` or `jnp.array`)
- Do not use `jit_forward_pass(x)` if `x.shape()` is expected to vary to avoid recompilation overhead

## picture of my cat
<img width="477" height="443" alt="zelda" src="https://github.com/user-attachments/assets/23a099cd-195d-48d7-b885-8baf84c4ff5e" />
