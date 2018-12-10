import numpy as np
from math import log10


class Neural_Network(object):
    def __init__(self,inputSize,hiddenLayerSize,outputSize,numberOfHidden):
        #define parameters according to dynamic parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenLayerSize
        self.outputSize = outputSize
        self.numberOfHidden = numberOfHidden
        self.weight_matrix_dict = {}
        self.weight_name_list = []
        np.random.seed(1) # take the same random variables for each initialized matrix

        for i in range(self.numberOfHidden + 1):
            weight_matrix_name = "W" + str(i + 1)
            self.weight_name_list.append(weight_matrix_name)

        # define weights random initially
        for i in range(len(self.weight_name_list)):
            weight_matrix_name = self.weight_name_list[i]
            if i == 0:
                self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.inputSize, self.hiddenSize)
                #self.W1 = self.weight_matrix_dict[weight_matrix_name]
            elif i != (len(self.weight_name_list) - 1):
                self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.hiddenSize, self.hiddenSize)
            else:
                self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.hiddenSize, self.outputSize)
                #self.W2 = self.weight_matrix_dict[weight_matrix_name]

        for weight_matrix_name in self.weight_name_list:
            print(weight_matrix_name, self.weight_matrix_dict[weight_matrix_name])



    def forwardPropagation(self, X):
        self.z_result = 0
        self.sigmoided_z_result = 0
        # for each matrix in matrix dictionary, iterate forward propagation
        for weight_matrix in self.weight_name_list:
            if weight_matrix == "W1":
                self.z_result = np.dot(X, self.weight_matrix_dict[weight_matrix]) # forward propagation
                #print("\nz", self.z_result)
                self.sigmoided_z_result = self.sigmoid(self.z_result)  # activation function
                #print("\nsigmoided_z", self.sigmoided_z_result)
            else:
                self.z_result = np.dot(self.sigmoided_z_result, self.weight_matrix_dict[weight_matrix])  # forward propagation
                #print("\nz", self.z_result)
                self.sigmoided_z_result = self.sigmoid(self.z_result)  # activation function
                #print("\nsigmoided_z", self.sigmoided_z_result)
        #return self.sigmoided_z_result

        """
        # forward propagation through our network
        self.z_1 = np.dot(X, self.W1)  # dot product of X (input) and first set of 3x2 weights
        self.sigmoided_z_1 = self.sigmoid(self.z_1)  # activation function
        self.z_2 = np.dot(self.sigmoided_z_1, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        self.sigmoided_z_2 = self.sigmoid(self.z_2)  # final activation function
        return self.sigmoided_z_2
        """
        #print("z result suan: ", self.sigmoided_z_result)

        # call softmax function for each row of output 2-d numpy array(for each image actually)
        softmax_results = np.apply_along_axis(self.softmax, 1, self.sigmoided_z_result)
        print("\n\nsoftmax result: ", softmax_results)
        return softmax_results

    def backwardPropagation(self,softmax_results,expected_output):
        #self.output_error_results = self.cross_entropy(softmax_results,expected_output)
        print("\npred Cross: ", softmax_results)
        print("\nexpect Cross: ", expected_output)
        self.cross_entropy_results = np.apply_along_axis(self.cross_entropy, 1, softmax_results, expected_output)

        sument = 0
        print("\ncross_entropy_results: \n")
        for i in self.cross_entropy_results:
            sument += i
            print(i)

        print("sument: ", sument / len(self.cross_entropy_results))

        print("\ncross_entropy_results: \n",self.cross_entropy_results)

        #self.cross_entropy(softmax_results, expected_output)
        #print(self.output_error_results)


    def trainModel(self, inputs, expectedOutputs):
        predicted_outputs = self.forwardPropagation(inputs)
        self.backwardPropagation(predicted_outputs, expectedOutputs)


    # activation sigmoid function
    def sigmoid(self, z):
        result = 1/(1+np.exp(-z))
        return result

    # derivative of sigmoid function
    def derivate_the_sigmoid(self, z):
        result = z * (1 - z)
        return result

    def softmax(self, outputs):
        exps = [np.exp(i) for i in outputs] # take exp value for each element of outputs
        exps = np.asarray(exps)
        sumOfExps = sum(exps) # take some of exponentials
        softmax = [j/sumOfExps for j in exps] # for each element of exps, divide element to sum of exps
        softmax = np.asarray(softmax)
        return softmax

    #soft_max = softmax(x)
    def derivate_the_softmax(self,soft_max):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = soft_max.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def cross_entropy(self, predictOutputs, expectedOutput):
        errorSum = 0
        for i in range(len(predictOutputs)):
            errorSum += (expectedOutput[i] * log10(predictOutputs[i]) + (1 - expectedOutput[i]) * log10(1 - predictOutputs[i]))

        result = - errorSum
        sum_of_vector = sum(result)
        output_node_number = len(expectedOutput[0])
        mean_of_errors = sum_of_vector / output_node_number
        return mean_of_errors

    def derivate_the_cross_entropy(self, predict, actual):
        return - actual / predict



    """
    def cross_entropy(self, predicted, expected):
        print("\n\npredicted: ", predicted)
        print("\n\nexpected: ", expected)
        print("\n\nDifference: ", predicted - expected)


        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.


        m = expected.shape[0]
        print("M: ", m)

        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        log_likelihood = -np.log(predicted[range(m), expected])
        print(log_likelihood)

        loss = np.sum(log_likelihood) / m
        return loss
    """