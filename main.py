from read_file import read_mat,expectedOutputs
from single_layer import normalize,single_layer_network
from neural_network import Neural_Network
import numpy as np



def programWorkStation():


    image_values = read_mat("train.mat")[0]
    normalized_images = normalize(image_values)
    expected_classes = read_mat("train.mat")[1]
    expected_outputs = expectedOutputs(expected_classes)

    X = normalized_images

    size_of_one_image = len(normalized_images[0])
    size_of_input = size_of_one_image
    hidden_node_number = 200
    hidden_layer_number = 0
    size_of_output = 5

    Beauty_Neural_Network = Neural_Network(size_of_input, hidden_node_number, size_of_output, hidden_layer_number)
    Beauty_Neural_Network.trainModel(X, expected_outputs)



    """
    # X = (hours sleeping, hours studying), y = score on test
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)

    # scale units
    X = X / 10  # maximum of X array
    y = y / 100  # max test score is 100

    size_of_input = 2
    hidden_node_number = 3
    hidden_layer_number = 1
    size_of_output = 1
    Beauty_Neural_Network = Neural_Network(size_of_input, hidden_node_number, size_of_output, hidden_layer_number)
    Beauty_Neural_Network.trainModel(X, y)
    """

programWorkStation()


