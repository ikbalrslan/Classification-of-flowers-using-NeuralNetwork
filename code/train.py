from read_file import read_mat,expectedOutputs,normalize
from neural_network import Neural_Network
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import argparse

"""
This function stands for drawing plots of the model accuracy or loss values
"""
def plot(pkl_name_create, train_accuracy_list, validation_accuracy_list, epoch_list):

    x1 = train_accuracy_list
    x2 = validation_accuracy_list

    plt.plot(epoch_list, x1, color='green')
    plt.plot(epoch_list, x2, color='orange')

    plt.legend(['train', 'validation'], loc='upper right')

    plt.title("Optimum Train and Validation Accuracy with 2 Hidden Layer")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy");

    figure_name = "Optimum_Train_and_Validation_Accuracy_with_2_Hidden_Layer" + ".png"

    plt.savefig(figure_name)
    plt.show()

"""
epochProcess takes some parameters such as epoch size, batch size, learning rate, neural network etc.
while neural network is creating, each layer node number and hidden layer number of neural network can 
be changed dynamically. I write the code dynamically, so neural network creates weights and connections 
while using the same function with given parameters.  
"""
def epochProcess(epoch_size, batch_size, learning_rate,  NN, size_den_inp, deneme_input, deneme_expected):
    validation_image_values = read_mat("validation.mat")[0] # images
    normalized_images_validation = normalize(validation_image_values)  # normalized images
    expected_classes_validation = read_mat("validation.mat")[1]  # expected flower types
    expected_outputs_validation = expectedOutputs(expected_classes_validation)  # flatten outputs
    X_validation = normalized_images_validation  # normalized input images
    size_of_validation = len(X_validation)
    loss_error_list = []
    epoch_list = []
    train_accuracy_list = []
    validation_accuracy_list = []
    for i in range(epoch_size):
        print("\nepoch",i+1,"-->")
        NN.hit = 0
        NN.total_loss_value = 0
        for j in range(0, size_den_inp, batch_size):
            batch_input =  deneme_input[j:j + batch_size]
            batch_expect = deneme_expected[j:j + batch_size]
            hit = NN.trainModel(batch_input, batch_expect, learning_rate, batch_size)

        print("train hit: ", NN.hit, ", Accuracy: ", NN.hit / size_den_inp * 100)
        train_accuracy_list.append(NN.hit / size_den_inp * 100)
        print("train loss: ", NN.total_loss_value )
        loss_error_list.append(NN.total_loss_value )
        epoch_list.append(i)

        # print("validation size: ", size_of_validation)
        predicted_outputs_validation = NN.forwardPropagation(X_validation, expected_outputs_validation)
        valid_hit = NN.hit_count(predicted_outputs_validation, expected_outputs_validation)
        print("\nvalidation hit: ", valid_hit, "accuracy: ", valid_hit / size_of_validation * 100)
        validation_accuracy_list.append(valid_hit / size_of_validation * 100)

    loss_error_list = np.asarray(loss_error_list)
    print("number of layer: ", NN.numberOfHidden, "layer size: ", NN.hiddenSize, "learning rate: ", learning_rate,
          "batch size: ", batch_size)
    pkl_name_create = str(NN.numberOfHidden) + "_" + str(NN.hiddenSize) + "_" + str(learning_rate) + "_" + str(batch_size)
    pkl_name_create += "_" + str(epoch_size)
    # Save to file in the current working directory
    pkl_file = pkl_name_create + ".pkl"
    with open(pkl_file, 'wb') as file:
        pickle.dump(NN, file, pickle.HIGHEST_PROTOCOL)

    #plot(pkl_name_create, train_accuracy_list, validation_accuracy_list, epoch_list)

"""
programWorkStation is the baseline of the assignment. Calls specific functions.
"""
def programWorkStation(train_file):

    image_values = read_mat(train_file)[0] # images
    normalized_images = normalize(image_values) # normalized images
    expected_classes = read_mat(train_file)[1] # expected flower types
    expected_outputs = expectedOutputs(expected_classes) # flatten outputs
    X = normalized_images #normalized input images
    size_of_one_image = len(normalized_images[0])
    size_of_input = size_of_one_image

    # parameters of neural network
    hidden_node_number = 100
    hidden_layer_number = 2
    size_of_output = 5
    learning_rate = 0.005
    epoch_size = 300
    batch_size = 20

    # neural network object is created here.
    Beauty_Neural_Network = Neural_Network(size_of_input, hidden_node_number, size_of_output, hidden_layer_number)

    deneme_input = X
    size_den_inp = len(deneme_input)
    den_expected = expected_outputs

    # run the code according to epoch and batch sizes.
    epochProcess(epoch_size, batch_size, learning_rate, Beauty_Neural_Network, size_den_inp, deneme_input, den_expected)


parser = argparse.ArgumentParser(description='train the model')
parser.add_argument('train_data_file', type=argparse.FileType('rb'))
args = parser.parse_args()

# program station is called here.
programWorkStation(args.train_data_file)



