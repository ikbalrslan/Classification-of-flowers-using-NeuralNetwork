from read_file import read_mat
from single_layer import normalize,single_layer_network

def programWorkStation():
    image_values = read_mat("train.mat")[0]
    single_layer_network(image_values)

programWorkStation()

