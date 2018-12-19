from read_file import read_mat,expectedOutputs,normalize
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
    hidden_node_number = 100
    hidden_layer_number = 2
    size_of_output = 5
    learning_rate = 0.02
    epoch_size = 1000
    batch_size = 100

    Beauty_Neural_Network = Neural_Network(size_of_input, hidden_node_number, size_of_output, hidden_layer_number)
    #Beauty_Neural_Network.trainModel(X, expected_outputs, learning_rate)
    #print("input: ", X)
    #Beauty_Neural_Network.trainModel(X, expected_outputs, learning_rate, batch_size)

    deneme_input = X
    size_den_inp = len(deneme_input)
    deneme_expected = expected_outputs
    size_den_exp = len(deneme_expected)

    #print(X[0].reshape(1,-1))
    #print(X[0])


    for i in range(epoch_size):
        print("\nepoch",i+1,"-->")
        Beauty_Neural_Network.hit = 0
        Beauty_Neural_Network.total_loss_value = 0
        for j in range(0, size_den_inp, batch_size):
            batch_input =  deneme_input[j:j + batch_size]
            batch_expect = deneme_expected[j:j + batch_size]
            #print("input: ", batch_input[-1].reshape(1,-1).shape)
            #print("expect: ", batch_expect)
            hit = Beauty_Neural_Network.trainModel(batch_input, batch_expect, learning_rate, batch_size)
        print("hit: ", hit, ", Accuracy: ",hit / 3000 * 100)
        print("loss: ", Beauty_Neural_Network.total_loss_value)


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


