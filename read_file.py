#!python
#!/usr/bin/env python
from scipy.io import loadmat, whosmat
import numpy as np


def read_mat(file_name):
    loader = loadmat(file_name)
    expected_outputs = loader['y'][0]
    expected_outputs = np.array(expected_outputs, dtype=float)
    image_values =  loader['x']
    image_values = np.array(image_values, dtype=float)
    #print(whosmat(file_name))
    return [image_values, expected_outputs]
