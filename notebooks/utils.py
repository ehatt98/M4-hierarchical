import arviz as az
import matplotlib.pyplot as plt
import corner
import jax 
import numpy as np
import jax.numpy as jnp

def unnorm(value, parameter_name, normalisation_constants):
    
    min_ = normalisation_constants[parameter_name][0]
    
    max_ = normalisation_constants[parameter_name][1]

    return value*(max_ - min_) + min_

def norm(value, parameter_name, normalisation_constants):
    
    min_ = normalisation_constants[parameter_name][0]
    
    max_ = normalisation_constants[parameter_name][1]

    return (value - min_)/(max_ - min_)




def unnorm_matrix(values, normalisation_constants):

    mins = normalisation_constants[0]

    maxes = normalisation_constants[1]

    return values*(maxes - mins) + mins

def norm_matrix(values, normalisation_constants):

    mins = normalisation_constants[0]

    maxes = normalisation_constants[1]

    return (values - mins)/(maxes - mins)

    
    