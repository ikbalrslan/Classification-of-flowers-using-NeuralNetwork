import numpy as np

class Neural_Network(object):
    def __init__(self,inputSize,hiddenLayerSize,outputSize,numberOfHidden):
        #define parameters according to dynamic parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenLayerSize
        self.outputSize = outputSize
        self.numberOfHidden = numberOfHidden
        self.weight_matrix_dict = {}
        self.weight_name_list = []

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

        #print("\nstatik W1: ", self.W1)
        #print("\nstatik W2: ", self.W2)

    def forward(self, X):
        self.z_result = 0
        self.sigmoided_z_result = 0
        # for each matrix in matrix dictionary, iterate forward propagation
        for weight_matrix in self.weight_name_list:
            if weight_matrix == "W1":
                self.z_result = np.dot(X, self.weight_matrix_dict[weight_matrix]) # forward propagation
                print("\nz", self.z_result)
                self.sigmoided_z_result = self.sigmoid(self.z_result)  # activation function
                print("\nsigmoided_z", self.sigmoided_z_result)
            else:
                self.z_result = np.dot(self.sigmoided_z_result, self.weight_matrix_dict[weight_matrix])  # forward propagation
                print("\nz", self.z_result)
                self.sigmoided_z_result = self.sigmoid(self.z_result)  # activation function
                print("\nsigmoided_z", self.sigmoided_z_result)
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
        return softmax_results

    # activation sigmoid function
    def sigmoid(self, z):
        result = 1/(1+np.exp(-z))
        return result

    def softmax(self, outputs):
        exps = [np.exp(i) for i in outputs] # take exp value for each element of outputs
        exps = np.asarray(exps)
        sumOfExps = sum(exps) # take some of exponentials
        softmax = [j/sumOfExps for j in exps] # for each element of exps, divide element to sum of exps
        softmax = np.asarray(softmax)
        return softmax


    # derivative of sigmoid function
    def sigmoidDerivation(self, z):
        result = z * (1 - z)
        return result

