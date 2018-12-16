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

def expectedOutputs(expected_classes):
    expected_output = []
    for i in expected_classes:
        temp = np.zeros(5)
        temp[int(i)] = 1
        expected_output.append(temp)
    expected_output = np.array(expected_output, dtype=float)
    return expected_output

def normalize(image_values):
    size_of_train_images = len(image_values)
    #random_array = np.random.uniform(low=0.0, high=1.0, size=(size_of_train_images,size_of_image))

    #print(image_values[size_of_train_images-1])
    normalized_images = np.divide(image_values, 255)
    #print(normalized_images[size_of_train_images-1])
    return normalized_images