from read_file import read_mat,expectedOutputs
from single_layer import normalize,single_layer_network
from neural_network import Neural_Network
import numpy as np



def programWorkStation():
    image_values = read_mat("train.mat")[0]
    normalized_images = normalize(image_values)
    expected_classes = read_mat("train.mat")[1]
    expected_outputs = expectedOutputs(expected_classes)
    #print(expected_outputs)

    #single_layer_network(image_values)

    ############################## TRY ####################################


    # X = (hours sleeping, hours studying), y = score on test
    X = normalized_images
    #y = np.array(([92], [86], [89]), dtype=float)
    
    y = expected_outputs / 5  # max test score is 100

    size_of_one_image = len(normalized_images[0])
    size_of_input = size_of_one_image
    hidden_node_number = 3
    hidden_layer_number = 1
    size_of_output = 5

    """
    # X = (hours sleeping, hours studying), y = score on test
    X = np.array(([2, 9], [5,8]), dtype=float)
    y = np.array(([92], [89]), dtype=float)

    # scale units
    X = X / np.amax(X, axis=0)  # maximum of X array
    y = y / 100  # max test score is 100
    """
    Beauty_Neural_Network = Neural_Network(size_of_input, hidden_node_number, size_of_output, hidden_layer_number)

    print("X: ", X)

    # defining our output
    #my_outputs = Beauty_Neural_Network.forwardPropagation(X)

    #print("\nPredicted Outputs: \n" + str(my_outputs))

    #soumOfResults = np.apply_along_axis(sum, 1, my_outputs)
    #print("\nsum of Outputs: \n" + str(soumOfResults))
    print("\nActual Output: \n" + str(expected_outputs))

    print("\nError Array: ", Beauty_Neural_Network.trainModel(X, expected_outputs))

programWorkStation()


