from read_file import read_mat
import numpy as np


def single_layer_network(image_values):
    normalized_images = normalize(image_values)
    #print(normalized_images)

def loss_function():
    print("loss")

def normalize(image_values):
    size_of_train_images = len(image_values)
    #random_array = np.random.uniform(low=0.0, high=1.0, size=(size_of_train_images,size_of_image))

    #print(image_values[size_of_train_images-1])
    normalized_images = np.divide(image_values, 255)
    #print(normalized_images[size_of_train_images-1])
    return normalized_images