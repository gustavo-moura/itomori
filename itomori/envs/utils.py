import numpy as np
import math
import yaml

def argmax(q_values):
    """
    Takes in a matrix of n*k q_values and returns the index
    of the item with the highest value for each row. 
    Breaks ties randomly.
    returns: vector of size n, where each item is the index of
    the highest value in q_values for each row.
    """
    # Generate a mask of the max values for each row
    mask = q_values == q_values.max(axis=1)[:, None]
    # Generate noise to be added to the ties
    r_noise = 1e-6*np.random.random(q_values.shape)
    # Get the argmax of the noisy masked values
    return np.argmax(r_noise*mask,axis=1)

def euclidean_distance(p1, p2):
    '''Calculate the Euclidean distance between two points.'''
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist

def load_yaml(file_path):
    '''Load a yaml file.'''
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
