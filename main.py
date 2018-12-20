from read_file import read_mat,expectedOutputs,normalize
from neural_network import Neural_Network
import pickle
import numpy as np

"""
epochProcess takes some parameters such as epoch size, batch size, learning rate, neural network etc.
while neural network is creating, each layer node number and hidden layer number of neural network can 
be changed dynamically. I write the code dynamically, so neural network creates weights and connections 
while using the same function with given parameters.  
"""
def epochProcess(epoch_size, batch_size, learning_rate,  NN, size_den_inp, deneme_input, deneme_expected):
    for i in range(epoch_size):
        print("\nepoch",i+1,"-->")
        NN.hit = 0
        NN.total_loss_value = 0
        for j in range(0, size_den_inp, batch_size):
            batch_input =  deneme_input[j:j + batch_size]
            batch_expect = deneme_expected[j:j + batch_size]
            #print("input: ", batch_input[-1].reshape(1,-1).shape)
            #print("expect: ", batch_expect)
            hit = NN.trainModel(batch_input, batch_expect, learning_rate, batch_size)

            # Save to file in the current working directory
            pkl_file = "model.pkl"
            with open(pkl_file, 'wb') as file:
                pickle.dump(NN, file)

        print("hit: ", hit, ", Accuracy: ",hit / 3000 * 100)
        print("loss: ", NN.total_loss_value)
"""
programWorkStation is the baseline of the assignment. Calls specific functions.
"""
def programWorkStation():

    image_values = read_mat("train.mat")[0] # images
    normalized_images = normalize(image_values) # normalized images
    expected_classes = read_mat("train.mat")[1] # expected flower types

    expected_outputs = expectedOutputs(expected_classes) # flatten outputs
    X = normalized_images #normalized input images

    size_of_one_image = len(normalized_images[0])
    size_of_input = size_of_one_image

    # parameters of neural network
    hidden_node_number = 100
    hidden_layer_number = 0
    size_of_output = 5
    learning_rate = 0.005
    epoch_size = 500
    batch_size = 20

    # neural network object is created here.
    Beauty_Neural_Network = Neural_Network(size_of_input, hidden_node_number, size_of_output, hidden_layer_number)

    deneme_input = X
    size_den_inp = len(deneme_input)
    den_expected = expected_outputs

    # run the code according to epoch and batch sizes.
    epochProcess(epoch_size, batch_size, learning_rate, Beauty_Neural_Network, size_den_inp, deneme_input, den_expected)


# program station is called here.
programWorkStation()



