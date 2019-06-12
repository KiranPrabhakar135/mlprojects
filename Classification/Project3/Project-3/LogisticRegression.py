import numpy as np
import DataProcessing as data
import math

number_of_classes = 10
number_of_features = np.array(data.mnist_training_data).shape[1]
training_data_length = np.array(data.mnist_training_data).shape[0]


def softmax(array):
    softmax_array = []
    for row in array:
        sum_e = 0
        arr = []
        for i in row:
            sum_e += math.exp(i)
        for i in row:
            arr.append(math.exp(i)/sum_e)
        softmax_array.append(arr)
    return np.array(softmax_array)
    # array -= np.max(array)
    # return np.exp(array) / np.sum(np.exp(array))


def sigmoidal_regression(epochs, learning_rate, training_data, one_hot_training_target):
    weight_matrix = np.zeros((number_of_features, number_of_classes))

    for i in range(epochs):
        output = np.dot(training_data, weight_matrix)  # 50k X 10
        softmax_output = softmax(output)
        cost = (np.subtract(softmax_output, one_hot_training_target))  # 50kX10
        gradient = (learning_rate/training_data_length)*np.dot(np.transpose(training_data),cost)
        weight_matrix = np.subtract(weight_matrix, gradient)
    return weight_matrix


def calculate_output(weights, testing_data):
    return softmax(np.dot(testing_data, weights))


def perform_logistic_regression_and_output_accuracy(weights):
    output = calculate_output(weights, data.mnist_testing_data)
    output = np.argmax(output, axis=1).transpose()
    accuracy, error = data.calculate_accuracy_and_error(output, data.mnist_testing_target)
    print("Logistic Regression: Mnist Accuracy: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(output, data.mnist_testing_target)
    data.create_heat_map(confusion_matrix)
    output = calculate_output(weights, data.usps_testing_data)
    output = np.argmax(output, axis=1).transpose()
    accuracy, error = data.calculate_accuracy_and_error(output, data.usps_testing_target)
    print("Logistic Regression: Usps accuracy: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(output, data.usps_testing_target)
    data.create_heat_map(confusion_matrix)


def calculate_weights_perform_logistic_regression_and_output_accuracy():
    weights = sigmoidal_regression(1000, 0.5, data.mnist_training_data, data.mnist_one_hot_vector_training_target)
    perform_logistic_regression_and_output_accuracy(weights)

#calculate_weights_perform_logistic_regression_and_output_accuracy()