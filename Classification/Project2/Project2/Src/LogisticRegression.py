import DataSet.DataProcessing as data
import numpy as np
import math
from matplotlib import pyplot as plt

training_accuracy_array = []
validation_accuracy_array = []
testing_accuracy_array = []
training_error_array = []
validation_error_array = []
testing_error_array = []


def clear_global_accuracy_arrays():
    training_accuracy_array.clear()
    validation_accuracy_array.clear()
    testing_accuracy_array.clear()


def clear_global_erms_arrays():
    training_error_array.clear()
    validation_error_array.clear()
    testing_error_array.clear()


def sigmoid(x):
    sigmoid_lambda = lambda t: math.exp(t) / (1 + math.exp(t))
    sigmoid_vectorized = np.vectorize(sigmoid_lambda)
    return sigmoid_vectorized(x)


def get_erms(model_generated_data, actual_data):
    temp_sum = 0.0
    counter = 0
    for i in range(0, len(model_generated_data)):
        temp_sum = temp_sum + math.pow((actual_data[i] - model_generated_data[i]), 2)
        if int(np.around(model_generated_data[i], 0)) == actual_data[i]:
            counter += 1
    accuracy = (float((counter * 100)) / float(len(model_generated_data)))
    return accuracy, math.sqrt(temp_sum / len(model_generated_data))


def calculate_logistic_output(data, weight_matrix):
    result = sigmoid(np.dot(data, weight_matrix))
    result = result <= 0.5
    result = result.astype(int)
    return result


def perform_logistic_regression(training_data, training_target, lambda_value, alpha, epochs):
    # training_data = np.array(training_data)
    # additional_column = np.ones((training_data.shape[0], 1), dtype= int)
    # training_data = np.append(additional_column, training_data, axis=1)

    weight_matrix = np.zeros((training_data.shape[1], 1))
    data_length = training_data.shape[0]

    for i in range(epochs):
        result = np.dot(training_data, weight_matrix)
        result = sigmoid(result)
        regularized_weights = np.dot(lambda_value, weight_matrix)
        error = np.subtract(result, np.array(training_target).reshape(-1, 1))
        gradient = (1 / data_length) * np.dot(training_data.transpose(), error)
        regularized_gradient = gradient + (1 / data_length) * regularized_weights
        regularized_gradient = alpha * regularized_gradient
        weight_matrix = np.subtract(weight_matrix, regularized_gradient)
    return weight_matrix


def con_hum_logistic_regression(lambda_value, alpha, epochs):
    weights = perform_logistic_regression(data.concatenated_human_training_data,
                                          data.concatenated_human_training_target_data, lambda_value, alpha, epochs)
    # additional_column_training = np.ones((data.concatenated_human_training_data.shape[0], 1), dtype=int)
    # updated_training = np.append(additional_column_training, data.concatenated_human_training_data, axis=1)
    # updated_validation = np.append(np.ones((data.concatenated_human_validation_data.shape[0], 1), dtype=int), data.concatenated_human_validation_data, axis=1)
    # updated_testing = np.append(np.ones((data.concatenated_human_testing_data.shape[0], 1), dtype=int), data.concatenated_human_testing_data, axis=1)
    training_output = calculate_logistic_output(data.concatenated_human_training_data, weights)
    validation_output = calculate_logistic_output(data.concatenated_human_validation_data, weights)
    testing_output = calculate_logistic_output(data.concatenated_human_testing_data, weights)
    training_accuracy, training_error = get_erms(np.array(training_output)
                                                 , np.array(data.concatenated_human_training_target_data))

    validation_accuracy, validation_error = get_erms(np.array(validation_output)
                                                     , np.array(data.concatenated_human_validation_target_data))

    testing_accuracy, testing_error = get_erms(np.array(testing_output)
                                               , np.array(data.concatenated_human_testing_target_data))

    training_accuracy_array.append(training_accuracy)
    validation_accuracy_array.append(validation_accuracy)
    testing_accuracy_array.append(testing_accuracy)
    training_error_array.append(training_error)
    validation_error_array.append(validation_error)
    testing_error_array.append(testing_error)


def sub_hum_logistic_regression(lambda_value, alpha, epochs):
    weights = perform_logistic_regression(data.subtracted_human_training_data,
                                          data.subtracted_human_training_target_data, lambda_value, alpha, epochs)
    training_output = calculate_logistic_output(data.subtracted_human_training_data, weights)
    validation_output = calculate_logistic_output(data.subtracted_human_validation_data, weights)
    testing_output = calculate_logistic_output(data.subtracted_human_testing_data, weights)
    training_accuracy, training_error = get_erms(np.array(training_output)
                                                 , np.array(data.subtracted_human_training_target_data))

    validation_accuracy, validation_error = get_erms(np.array(validation_output)
                                                     , np.array(data.subtracted_human_validation_target_data))

    testing_accuracy, testing_error = get_erms(np.array(testing_output)
                                               , np.array(data.subtracted_human_testing_target_data))

    training_accuracy_array.append(training_accuracy)
    validation_accuracy_array.append(validation_accuracy)
    testing_accuracy_array.append(testing_accuracy)
    training_error_array.append(training_error)
    validation_error_array.append(validation_error)
    testing_error_array.append(testing_error)


def con_gsc_logistic_regression(lambda_value, alpha, epochs):
    weights = perform_logistic_regression(data.concatenated_gsc_training_data,
                                          data.concatenated_gsc_training_target_data, lambda_value, alpha, epochs)
    training_output = calculate_logistic_output(data.concatenated_gsc_training_data, weights)
    validation_output = calculate_logistic_output(data.concatenated_gsc_validation_data, weights)
    testing_output = calculate_logistic_output(data.concatenated_gsc_testing_data, weights)
    training_accuracy, training_error = get_erms(np.array(training_output)
                                                 , np.array(data.concatenated_gsc_training_target_data))

    validation_accuracy, validation_error = get_erms(np.array(validation_output)
                                                     , np.array(data.concatenated_gsc_validation_target_data))

    testing_accuracy, testing_error = get_erms(np.array(testing_output)
                                               , np.array(data.concatenated_gsc_testing_target_data))

    training_accuracy_array.append(training_accuracy)
    validation_accuracy_array.append(validation_accuracy)
    testing_accuracy_array.append(testing_accuracy)
    training_error_array.append(training_error)
    validation_error_array.append(validation_error)
    testing_error_array.append(testing_error)


def sub_gsc_logistic_regression(lambda_value, alpha, epochs):
    weights = perform_logistic_regression(data.subtracted_gsc_training_data, data.subtracted_gsc_training_target_data,
                                          lambda_value, alpha, epochs)
    training_output = calculate_logistic_output(data.subtracted_gsc_training_data, weights)
    validation_output = calculate_logistic_output(data.subtracted_gsc_validation_data, weights)
    testing_output = calculate_logistic_output(data.subtracted_gsc_testing_data, weights)
    training_accuracy, training_error = get_erms(np.array(training_output)
                                                 , np.array(data.subtracted_gsc_training_target_data))

    validation_accuracy, validation_error = get_erms(np.array(validation_output)
                                                     , np.array(data.subtracted_gsc_validation_target_data))

    testing_accuracy, testing_error = get_erms(np.array(testing_output)
                                               , np.array(data.subtracted_gsc_testing_target_data))

    training_accuracy_array.append(training_accuracy)
    validation_accuracy_array.append(validation_accuracy)
    testing_accuracy_array.append(testing_accuracy)
    training_error_array.append(training_error)
    validation_error_array.append(validation_error)
    testing_error_array.append(testing_error)


def plot_alpha_vs_accuracy(alpha_array):
    plt.figure()
    plt.plot(alpha_array, training_accuracy_array, label="training")
    plt.plot(alpha_array, validation_accuracy_array, label="validation")
    plt.plot(alpha_array, testing_accuracy_array, label="testing")
    plt.legend()
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate Vs Accuracy')
    plt.show()
    clear_global_accuracy_arrays()


def plot_alpha_vs_erms(alpha_array):
    plt.figure()
    plt.plot(alpha_array, training_error_array, label="training")
    plt.plot(alpha_array, validation_error_array, label="validation")
    plt.plot(alpha_array, testing_error_array, label="testing")
    plt.legend()
    plt.xlabel('Learning Rate')
    plt.ylabel('Erms')
    plt.title('Learning Rate Vs ERMS')
    plt.show()
    clear_global_erms_arrays()


def con_hum_logistic_regression_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        con_hum_logistic_regression(2, alpha, 2000)
    print("Concatenated Human data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Concatenated Human data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array))+ " }")


def sub_hum_logistic_regression_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        sub_hum_logistic_regression(2, alpha, 1000)
    print("Subtracted Human data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Subtracted Human data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def con_gsc_logistic_regression_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        con_gsc_logistic_regression(2, alpha, 1000)
    print("Concatenated GSC data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Concatenated GSC data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def sub_gsc_logistic_regression_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        sub_gsc_logistic_regression(2, alpha, 1000)
    print("Subtracted GSC data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Subtracted GSC data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")