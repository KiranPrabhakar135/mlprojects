import DataSet.DataProcessing as data
import numpy as np
from sklearn.cluster import KMeans
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


def get_big_sigma_matrix(data):
    big_sigma_matrix = np.identity(data.shape[1])
    for i in range(big_sigma_matrix.shape[0]):
        ith_column_values = np.array(data.iloc[:, i:i+1])
        variance = np.var(np.array(ith_column_values))
        big_sigma_matrix[i][i] = variance
    return np.dot(0.2, big_sigma_matrix)


def calculate_basis_function(x, big_sigma_inverse, mean):
    x_minus_mean = np.subtract(x, mean)
    return np.exp(-0.5 * np.dot(np.dot(x_minus_mean,
                                       big_sigma_inverse), np.transpose(x_minus_mean)))


def calculate_phi_matrix(data_input, big_sigma_inverse, means):
    data_input = np.array(data_input)
    input_data_length = len(data_input)
    means_length = len(means)
    big_phi_matrix = np.zeros((len(data_input), len(means)))

    for i in range(input_data_length):
        for j in range(means_length):
            big_phi_matrix[i][j] = calculate_basis_function(np.array(data_input)[i], big_sigma_inverse, means[j])
    return np.array(big_phi_matrix)


def calculate_output(phi_matrix, weight_matrix):
    return np.dot(phi_matrix, weight_matrix)


def get_erms(model_generated_data, actual_data):
    temp_sum = 0.0
    counter = 0
    for i in range(0, len(model_generated_data)):
        temp_sum = temp_sum + math.pow((actual_data[i] - model_generated_data[i]), 2)
        if int(np.around(model_generated_data[i], 0)) == actual_data[i]:
            counter += 1
    accuracy = (float((counter*100)) / float(len(model_generated_data)))
    return accuracy, math.sqrt(temp_sum / len(model_generated_data))


def calculate_clusters_and_phi(m_value, training_data, big_sigma_inverse):
    training_clusters = KMeans(m_value, random_state=0).fit(training_data)
    training_means = training_clusters.cluster_centers_
    return calculate_phi_matrix(training_data, big_sigma_inverse, training_means)


def perform_linear_regression(training_target, training_phi, lambda_value, alpha, epochs):
    weight_matrix = np.zeros((training_phi.shape[1], 1))
    data_length = training_phi.shape[0]

    for i in range(epochs):
        result = np.dot(training_phi, weight_matrix)
        regularized_weights = np.dot(lambda_value, weight_matrix)
        error = np.subtract(result, np.array(training_target).reshape(-1, 1))
        gradient = (1/data_length) * np.dot(error.transpose(), training_phi)
        regularized_gradient = gradient.transpose() + regularized_weights
        regularized_gradient = alpha * regularized_gradient
        weight_matrix = np.subtract(weight_matrix, regularized_gradient)

    return weight_matrix


def con_hum_linear_regression(m_value, lambda_value, alpha, epochs):
    big_sigma = get_big_sigma_matrix(data.total_concatenated_human_data)
    con_hum_big_sigma_inverse = np.linalg.inv(big_sigma)
    con_hum_training_phi = calculate_clusters_and_phi(m_value, data.concatenated_human_training_data, con_hum_big_sigma_inverse)
    con_hum_validation_phi = calculate_clusters_and_phi(m_value, data.concatenated_human_validation_data, con_hum_big_sigma_inverse)
    con_hum_testing_phi = calculate_clusters_and_phi(m_value, data.concatenated_human_testing_data, con_hum_big_sigma_inverse)

    con_hum_weights = perform_linear_regression(data.concatenated_human_training_target_data,
                                                                con_hum_training_phi,
                                                                lambda_value=lambda_value,
                                                                alpha= alpha,
                                                                epochs=epochs)

    training_output = calculate_output(con_hum_training_phi, con_hum_weights)
    validation_output = calculate_output(con_hum_validation_phi, con_hum_weights)
    testing_output = calculate_output(con_hum_testing_phi, con_hum_weights)
    return training_output, validation_output, testing_output


def sub_hum_linear_regression(m_value, lambda_value, alpha, epochs):
    big_sigma = get_big_sigma_matrix(data.total_subtracted_human_data)
    sub_hum_big_sigma_inverse = np.linalg.inv(big_sigma)
    sub_hum_training_phi = calculate_clusters_and_phi(m_value, data.subtracted_human_training_data, sub_hum_big_sigma_inverse)
    sub_hum_validation_phi = calculate_clusters_and_phi(m_value, data.subtracted_human_validation_data, sub_hum_big_sigma_inverse)
    sub_hum_testing_phi = calculate_clusters_and_phi(m_value, data.subtracted_human_testing_data, sub_hum_big_sigma_inverse)

    sub_hum_weights = perform_linear_regression(data.subtracted_human_training_target_data,
                                                 sub_hum_training_phi,
                                                 lambda_value=lambda_value,
                                                 alpha= alpha,
                                                 epochs=epochs)

    training_output = calculate_output(sub_hum_training_phi, sub_hum_weights)
    validation_output = calculate_output(sub_hum_validation_phi, sub_hum_weights)
    testing_output = calculate_output(sub_hum_testing_phi, sub_hum_weights)
    return training_output, validation_output, testing_output


def con_gsc_linear_regression(m_value, lambda_value, alpha, epochs):
    # big_sigma = get_big_sigma_matrix(data.total_concatenated_gsc_data)
    # con_gsc_big_sigma_inverse = np.linalg.inv(big_sigma)
    # con_gsc_training_phi = calculate_clusters_and_phi(m_value, data.concatenated_gsc_training_data,
    #                                                   con_gsc_big_sigma_inverse)
    # con_gsc_validation_phi = calculate_clusters_and_phi(m_value, data.concatenated_gsc_validation_data,
    #                                                     con_gsc_big_sigma_inverse)
    # con_gsc_testing_phi = calculate_clusters_and_phi(m_value, data.concatenated_gsc_testing_data,
    #                                                  con_gsc_big_sigma_inverse)

    con_gsc_weights = perform_linear_regression(data.concatenated_gsc_training_target_data,
                                                data.concatenated_gsc_training_data,
                                                lambda_value=lambda_value,
                                                alpha=alpha,
                                                epochs=epochs)

    training_output = calculate_output(data.concatenated_gsc_training_data, con_gsc_weights)
    validation_output = calculate_output(data.concatenated_gsc_validation_data, con_gsc_weights)
    testing_output = calculate_output(data.concatenated_gsc_testing_data, con_gsc_weights)
    return training_output, validation_output, testing_output


def sub_gsc_linear_regression(m_value, lambda_value, alpha, epochs):
    # big_sigma = get_big_sigma_matrix(data.total_subtracted_gsc_data)
    # sub_gsc_big_sigma_inverse = np.linalg.inv(big_sigma)
    # sub_gsc_training_phi = calculate_clusters_and_phi(m_value, data.subtracted_gsc_training_data,
    #                                                   sub_gsc_big_sigma_inverse)
    # sub_gsc_validation_phi = calculate_clusters_and_phi(m_value, data.subtracted_gsc_validation_data,
    #                                                     sub_gsc_big_sigma_inverse)
    # sub_gsc_testing_phi = calculate_clusters_and_phi(m_value, data.subtracted_gsc_testing_data,
    #                                                  sub_gsc_big_sigma_inverse)

    sub_gsc_weights = perform_linear_regression(data.subtracted_gsc_training_target_data,
                                                data.subtracted_gsc_training_data,
                                                lambda_value=lambda_value,
                                                alpha=alpha,
                                                epochs=epochs)

    training_output = calculate_output( data.subtracted_gsc_training_data, sub_gsc_weights)
    validation_output = calculate_output(data.subtracted_gsc_validation_data, sub_gsc_weights)
    testing_output = calculate_output(data.subtracted_gsc_testing_data, sub_gsc_weights)
    return training_output, validation_output, testing_output


def con_hum_multiple_m_values(m_values):
    for m_value in m_values:
        con_human_result = con_hum_linear_regression(m_value, 2, 0.01, 1000)
        training_accuracy, training_error = get_erms(np.array(con_human_result[0])
                                                     , np.array(data.concatenated_human_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(con_human_result[1])
                                                         , np.array(data.concatenated_human_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(con_human_result[2])
                                                   , np.array(data.concatenated_human_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Concatenated Human data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " +str( max(testing_accuracy_array)) + " }")
    print("Concatenated Human data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def con_hum_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        con_human_result = con_hum_linear_regression(10, 2, alpha, 1000)
        training_accuracy, training_error = get_erms(np.array(con_human_result[0])
                                                     , np.array(data.concatenated_human_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(con_human_result[1])
                                                         , np.array(data.concatenated_human_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(con_human_result[2])
                                                   , np.array(data.concatenated_human_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Concatenated Human data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Concatenated Human data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def sub_hum_multiple_m_values(m_values):
    for m_value in m_values:
        sub_human_result = sub_hum_linear_regression(m_value, 2, 0.01, 1000)
        training_accuracy, training_error = get_erms(np.array(sub_human_result[0])
                                                     , np.array(data.subtracted_human_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(sub_human_result[1])
                                                         , np.array(data.subtracted_human_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(sub_human_result[2])
                                                   , np.array(data.subtracted_human_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Subtracted Human data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Subtracted Human data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def sub_hum_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        sub_human_result = sub_hum_linear_regression(10, 2, alpha, 1000)
        training_accuracy, training_error = get_erms(np.array(sub_human_result[0])
                                                     , np.array(data.subtracted_human_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(sub_human_result[1])
                                                         , np.array(data.subtracted_human_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(sub_human_result[2])
                                                   , np.array(data.subtracted_human_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Subtracted Human data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Subtracted Human data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def con_gsc_multiple_m_values(m_values):
    for m_value in m_values:
        con_gsc_result = con_gsc_linear_regression(m_value, 2, 0.01, 1000)
        training_accuracy, training_error = get_erms(np.array(con_gsc_result[0])
                                                     , np.array(data.concatenated_gsc_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(con_gsc_result[1])
                                                         , np.array(data.concatenated_gsc_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(con_gsc_result[2])
                                                   , np.array(data.concatenated_gsc_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Concatenated GSC data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array))+ " }")
    print("Concatenated GSC data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def con_gsc_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        con_gsc_result = con_gsc_linear_regression(10, 2, alpha, 1000)
        training_accuracy, training_error = get_erms(np.array(con_gsc_result[0])
                                                     , np.array(data.concatenated_gsc_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(con_gsc_result[1])
                                                         , np.array(data.concatenated_gsc_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(con_gsc_result[2])
                                                   , np.array(data.concatenated_gsc_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Concatenated GSC data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Concatenated GSC data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def sub_gsc_multiple_m_values(m_values):
    for m_value in m_values:
        sub_gsc_result = sub_gsc_linear_regression(m_value, 2, 0.01, 1000)
        training_accuracy, training_error = get_erms(np.array(sub_gsc_result[0])
                                                     , np.array(data.subtracted_gsc_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(sub_gsc_result[1])
                                                         , np.array(data.subtracted_gsc_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(sub_gsc_result[2])
                                                   , np.array(data.subtracted_gsc_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Subtracted GSC data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Subtracted GSC data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def sub_gsc_multiple_alpha(alpha_array):
    for alpha in alpha_array:
        sub_gsc_result = sub_gsc_linear_regression(10, 2, alpha, 1000)
        training_accuracy, training_error = get_erms(np.array(sub_gsc_result[0])
                                                     , np.array(data.subtracted_gsc_training_target_data))

        validation_accuracy, validation_error = get_erms(np.array(sub_gsc_result[1])
                                                         , np.array(data.subtracted_gsc_validation_target_data))

        testing_accuracy, testing_error = get_erms(np.array(sub_gsc_result[2])
                                                   , np.array(data.subtracted_gsc_testing_target_data))

        training_accuracy_array.append(training_accuracy)
        validation_accuracy_array.append(validation_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        training_error_array.append(training_error)
        validation_error_array.append(validation_error)
        testing_error_array.append(testing_error)
    print("Subtracted GSC data - { Training Accuracy: " + str(max(training_accuracy_array)) + " Validation Accuracy: " +
          str(max(validation_accuracy_array)) + " Testing Accuracy: " + str(max(testing_accuracy_array)) + " }")
    print("Subtracted GSC data - { Training Error: " + str(min(training_error_array)) + " Validation Error: " +
          str(min(validation_error_array)) + " Testing Error: " + str(min(testing_error_array)) + " }")


def plot_clusters_vs_accuracy(m_values):
    # training_array = list(map(int, training_accuracy_array))
    # validation_array = list(map(int, validation_accuracy_array))
    # testing_array = list(map(int, testing_accuracy_array))
    plt.figure()
    plt.plot(m_values, training_accuracy_array, label="training")
    plt.plot(m_values, validation_accuracy_array, label="validation")
    plt.plot(m_values, testing_accuracy_array, label="testing")
    plt.legend()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Accuracy')
    plt.title('Clusters Vs Accuracy')
    plt.show()
    clear_global_accuracy_arrays()


def plot_alpha_vs_accuracy(alpa_array):
    plt.figure()
    plt.plot(alpa_array, training_accuracy_array, label="training")
    plt.plot(alpa_array, validation_accuracy_array, label="validation")
    plt.plot(alpa_array, testing_accuracy_array, label="testing")
    plt.legend()
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate Vs Accuracy')
    plt.show()
    clear_global_accuracy_arrays()


def plot_clusters_vs_erms(m_values):
    plt.figure()
    plt.plot(m_values, training_error_array, label="training")
    plt.plot(m_values, validation_error_array, label="validation")
    plt.plot(m_values, testing_error_array, label="testing")
    plt.legend()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Erms')
    plt.title('Clusters Vs ERMS')
    plt.show()
    clear_global_erms_arrays()


def plot_alpha_vs_erms(alpa_array):
    plt.figure()
    plt.plot(alpa_array, training_error_array, label="training")
    plt.plot(alpa_array, validation_error_array, label="validation")
    plt.plot(alpa_array, testing_error_array, label="testing")
    plt.legend()
    plt.xlabel('Learning Rate')
    plt.ylabel('Erms')
    plt.title('Learning Rate Vs ERMS')
    plt.show()
    clear_global_erms_arrays()