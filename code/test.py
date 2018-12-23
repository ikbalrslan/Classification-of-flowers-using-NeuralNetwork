import pickle
from read_file import read_mat,expectedOutputs,normalize
import argparse

def test_accuracy(pickle_model, X, expected_outputs):
    #print("Single Layer Neural Network")
    size_of_test = len(X)
    print("test_size: ", size_of_test)
    predicted_outputs = pickle_model.forwardPropagation(X, expected_outputs)
    hit = pickle_model.hit_count(predicted_outputs, expected_outputs)
    print("test hit: ", hit)
    print("accuracy:", hit / size_of_test * 100)

def loadPickle(pkl_file):
    # Load from file
    with open(pkl_file, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('test_data_file', type=argparse.FileType('rb'))
parser.add_argument('model_file')
args = parser.parse_args()

pkl_file = "2_Hidden_Layer.pkl"
#model = loadPickle(pkl_file)
model = loadPickle(args.model_file)

image_values = read_mat(args.test_data_file)[0]  # images
normalized_images = normalize(image_values)  # normalized images
expected_classes = read_mat(args.test_data_file)[1]  # expected flower types

expected_outputs = expectedOutputs(expected_classes)  # flatten outputs
X = normalized_images  # normalized input images

test_accuracy(model, X, expected_outputs)