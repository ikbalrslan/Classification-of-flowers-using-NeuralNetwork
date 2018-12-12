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
        self.bias_name_list = []
        self.bias_dict = {}
        self.step_outputs = []

        np.random.seed(1) # take the same random variables for each initialized matrix

        for i in range(self.numberOfHidden + 1):
            weight_matrix_name = "W" + str(i + 1)
            bias_name = "b" + str(i + 1)
            self.weight_name_list.append(weight_matrix_name)
            self.bias_name_list.append(bias_name)

        # define weights random initially
        for i in range(len(self.weight_name_list)):
            weight_matrix_name = self.weight_name_list[i]
            bias_name = self.bias_name_list[i]
            if self.numberOfHidden != 0:
                if i == 0:
                    self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.inputSize, self.hiddenSize)
                    #self.W1 = self.weight_matrix_dict[weight_matrix_name]
                    self.bias_dict[bias_name] = np.zeros((1, self.hiddenSize))
                    #print("biasss:  ",self.bias_dict[bias_name])
                elif i != (len(self.weight_name_list) - 1):
                    self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.hiddenSize, self.hiddenSize)
                    self.bias_dict[bias_name] = np.zeros((1, self.hiddenSize))
                    #print("biasss:  ",self.bias_dict[bias_name])
                else:
                    self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.hiddenSize, self.outputSize)
                    #self.W2 = self.weight_matrix_dict[weight_matrix_name]
                    self.bias_dict[bias_name] = np.zeros((1, self.outputSize))
                    #print("biasss:  ",self.bias_dict[bias_name])
            else: # if there is no hidden layer. This means that this is a single layer
                self.weight_matrix_dict[weight_matrix_name] = np.random.randn(self.inputSize, self.outputSize)
                # self.W1 = self.weight_matrix_dict[weight_matrix_name]
                self.bias_dict[bias_name] = np.zeros((1, self.outputSize))
                # print("biasss:  ",self.bias_dict[bias_name])

        for weight_matrix_name in self.weight_name_list:
            print("\n",weight_matrix_name, self.weight_matrix_dict[weight_matrix_name])
        for bias_name in self.bias_name_list:
            print("\n",bias_name, self.bias_dict[bias_name])


    def forwardPropagation(self, X, expectedOutputs):
        print("\n\ninput: ", X)

        self.z_result = 0
        self.a_result = 0
        # for each matrix in matrix dictionary, iterate forward propagation
        weight_list_size = len(self.weight_name_list)
        for i in range(weight_list_size):
            wName = self.weight_name_list[i]
            bName = self.bias_name_list[i]

            #single layer neural network
            if weight_list_size == 1: # use softmax as activation function for output
                self.z_result = np.dot(X, self.weight_matrix_dict[wName]) + self.bias_dict[bName]  # forward propagation
                self.step_outputs.append(self.z_result)
            #multilayer neural network
            else:
                if wName == "W1":
                    self.z_result = np.dot(X, self.weight_matrix_dict[wName]) + self.bias_dict[bName] # forward propagation
                    self.step_outputs.append(self.z_result)
                    self.a_result = self.sigmoid(self.z_result)  # activation function
                    self.step_outputs.append(self.a_result)

                elif i == (weight_list_size - 1): # use softmax as activation function for output
                    self.z_result = np.dot(self.a_result, self.weight_matrix_dict[wName]) + self.bias_dict[bName]  # forward propagation
                    self.step_outputs.append(self.z_result)

                else:
                    self.z_result = np.dot(self.a_result, self.weight_matrix_dict[wName]) + self.bias_dict[bName] # forward propagation
                    self.step_outputs.append(self.z_result)
                    self.a_result = self.sigmoid(self.z_result)  # activation function
                    self.step_outputs.append(self.a_result)

        # call softmax function for each row of output 2-d numpy array(for each image actually)
        softmax_results = self.softmax(self.z_result)
        self.step_outputs.append(softmax_results)

        #print("toplamına bak: ", sum(softmax_results[16]))

        for i in range(len(self.step_outputs)):
            print("\nstep output =>",i, self.step_outputs[i])

        self.total_loss(softmax_results, expectedOutputs)
        return softmax_results

    # takes batch size times image and for each image; calculates probabilities of image outputs
    def softmax(self, all_image_outputs):
        softmax_list = []
        for i in all_image_outputs:
            exps = []
            for j in i:
                exps.append(np.exp(j))
            exps = np.asarray(exps)
            sumOfExps = sum(exps)  # take some of exponentials
            softmax = []
            for j in exps:
                softmax.append(j/sumOfExps)
            softmax = np.asarray(softmax)
            softmax_list.append(softmax)
        softmax_list = np.asarray(softmax_list)
        return softmax_list

    def trainModel(self, inputs, expectedOutputs):
        predicted_outputs = self.forwardPropagation(inputs,expectedOutputs)
        self.backwardPropagation(inputs, predicted_outputs, expectedOutputs)

    def backwardPropagation(self, X, y, o):
        print("none!!!!")

        """
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.derivate_the_sigmoid(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.derivate_the_sigmoid(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights
        """

    # activation sigmoid function
    def sigmoid(self, z):
        result = 1/(1+np.exp(-z))
        return result

    # derivative of sigmoid function
    def derivate_the_sigmoid(self, z):
        result = z * (1 - z)
        return result

    def total_loss(self, softmax_results, expected_output):
        print("\nsoftmax_results of total: ", softmax_results)
        print("\nexpect of total: ", expected_output)

        self.cross_entropy_results = self.cross_entropy(softmax_results,expected_output)

        loss_sum = 0
        #print("\ncross_entropy_results: \n")
        for i in self.cross_entropy_results:
            loss_sum += i
            #print(i)

        total_loss = loss_sum / len(self.cross_entropy_results)
        print("\ncross_entropy_results and size: \n", self.cross_entropy_results, len(self.cross_entropy_results))
        print("\nLoss: \n", total_loss)

        # self.cross_entropy(softmax_results, expected_output)  # print(self.output_error_results)

    # E = – ∑ ci . log(pi) + (1 – ci ). log(1 – pi)
    # this function returns error list. one index for each image(index is mean of output errors for each image)
    def cross_entropy(self, softmax_results, expected_output):
        output_node_number = len(expected_output[0])
        error_list = []
        for i in range(len(softmax_results)):
            errorSum = 0
            for j in range(output_node_number):
                pre_ = softmax_results[i][j]
                exp_ = expected_output[i][j]
                #print("\npre: ",pre_,"exp: ",exp_)
                errorSum += (exp_ * log10(pre_) + (1 - exp_) * log10(1 - pre_))
            result = - errorSum
            mean_of_errors = result / output_node_number
            error_list.append(mean_of_errors)
        error_list = np.asarray(error_list)
        return error_list

    def derivate_the_cross_entropy(self, predict, actual):
        return - actual / predict

    #soft_max = softmax(x)
    def derivate_the_softmax(self,soft_max):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = soft_max.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)




