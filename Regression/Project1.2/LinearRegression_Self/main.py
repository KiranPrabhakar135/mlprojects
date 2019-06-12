import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt

# Variables declaration
dataFile = "Querylevelnorm_X.csv"
targetFile = "Querylevelnorm_t.csv"
training_percent = 80
validation_percent = 10
testing_percent = 10
M = 60  # number of clusters
lambda_constant = 0.03


# Variables used to draw graph
M_array = [10, 20, 50, 60, 100, 200]
lambda_array = [0.01, 0.03, 0.05, 0.3, 0.5, 0.7]
Training_Erms_Array = []
Validation_Erms_Array = []
Testing_Erms_Array = []
Training_Accuracy_Array = []
Validation_Accuracy_Array = []
Testing_Accuracy_Array = []


# read the data from .csv files using pandas data frames.
def read_data_from_file(file_path, remove_zero_columns):
    data_frame = pd.read_csv(file_path, header=None)
    if remove_zero_columns:
        data_frame = data_frame.drop([5, 6, 7, 8, 9], axis=1)
    return np.array(data_frame.values)


# generic method to return the input and target data for training, testing and validation based on provided percentage
def get_sliced_data(raw_data, percentage, is_validation_data, is_testing_data):
    raw_data_count = raw_data.shape[0]
    required_rows_count = int(raw_data_count * (percentage * 0.01)) # 0.01 converts percent to decimal.
    # while slicing for validation data, we skip the data items which are taken for training
    if is_validation_data:
        validation_data_start_index = int(raw_data_count * (training_percent * 0.01)) + 1
        return np.array(raw_data[validation_data_start_index: (validation_data_start_index + required_rows_count), :])
    # while slicing for the validation data, we skip the data items which are taken for training and validation
    if is_testing_data:
        training_data_start_index = int(raw_data_count * ((training_percent + validation_percent) * 0.01)) + 1
        return np.array(raw_data[training_data_start_index:(training_data_start_index + required_rows_count), :])
    return np.array(raw_data[0:required_rows_count, :])


# We calculate only the variances in same feature and we ignore the variances w.r.t other feature.
# As our features are independent, we need not calculate variance w.r.t to other feature.
# So, we only calculate the variance along the diagonal of the big sigma matrix and make other values to '0' .
# As each feature is independent of other feature, we can consider the variance w.r.t to other feature as 0
def get_big_sigma_matrix(data):
    big_sigma_matrix = np.identity(data.shape[1])  # length of first data will give the number of columns which is 41
    for i in range(big_sigma_matrix.shape[0]):
        #  ":," will not slice in rows but will slice column wise
        ith_column_values = np.array(data[:, i:i+1]).flatten().tolist()  # slices all the elements in the ith column
        variance = np.var(np.array(ith_column_values))  # calculate the variance of ith feature.
        big_sigma_matrix[i][i] = variance
    # It can be any random value, which handles the issue of not getting a singular matrix and allows inversion
    return np.dot(200, big_sigma_matrix)


# For every element in the design matrix, the output of the basis function is computed using the below function
# Gaussian radial basis function is used in the below function
def calculate_basis_function(x, big_sigma_inverse, mean):
    x_minus_mean = np.subtract(x,mean)  #1X41 - 1X 41
    product = np.dot(np.transpose(x_minus_mean), big_sigma_inverse)  # 41X41 dot 41X1
    product = np.dot(product, x_minus_mean)  # 1X41 dot 41X1
    product = -0.5 * product  # 1X1
    return math.exp(product)


# The method returns the design matrix used for regression.
# The function takes the input data, covariance matrix and the means of all the clusters.
# The output of this will be (no.of data elements) X (number of basis functions used) in the current
# case, for training set, it is 55698 X M.
# The element in the ith row and jth column of  phi matrix is the output of gaussian radial basis function,
# which is calculated using jth mean in set of means of each cluster and ith input in input data
# Also the covariance matrix is used for calculation of Phi matrix
def calculate_phi_matrix(data, big_sigma, means):
    big_phi_matrix = np.zeros((len(data), len(means)))
    big_sigma_inverse = np.linalg.inv(big_sigma)
    for i in range(len(data)):
        for j in range(len(means)):
            big_phi_matrix[i][j] = calculate_basis_function(data[i], big_sigma_inverse, means[j])
    return np.array(big_phi_matrix)


# The weight matrix is calculated using the training phi matrix.
# For calculating the weight matrix we need training target which is multiplied with the operations performed on phi
# matrix. In our case, the dimensions of phi matrix will be 10X1
# Calculating the weight matrix plays a crucial role in the performing linear regression.
# The weight matrix created here is used in both SGD and also in closed from solutions.
def calculate_W_matrix(phi_matrix, lam):
    identity_matrix = np.identity(len(phi_matrix[0]))
    identity_matrix = lam * identity_matrix
    phi_transpose = np.transpose(phi_matrix)  # 10X55k
    phi_x_phi_transpose = np.matmul(phi_transpose, phi_matrix)  # 10X10
    i_plus_phiT_phi = np.add(identity_matrix, phi_x_phi_transpose)  # 10X10
    inv_i_plus_phiT_phi = np.linalg.inv(i_plus_phiT_phi)  # 10X10
    product_inv_transpose = np.matmul(inv_i_plus_phiT_phi,phi_transpose)  # 10X55k
    w_matrix = np.dot(product_inv_transpose,training_target)  # 10X1
    return w_matrix


# performs y = w*X to get the calculated output of the model
def calculate_output(phi_matrix, weight_matrix):
    return np.dot(phi_matrix, weight_matrix)


# Returns the frequencies of occurrence of the calculated outputs as dictionary of items Eg:{0:100, 1:200}
def get_frequencies(data):
    return dict(pd.Series(data).value_counts())


# Calculates how accurate the output when compared with target.
def calculate_accuracy(output, target):
    rounded_output = np.array(np.round(output, 0))
    # If the difference between rounded_output and target is zero, then it has matched to the output
    difference = np.subtract(rounded_output, target)
    frequencies = get_frequencies(difference.flatten())
    count_of_match = frequencies[0]
    return (float(count_of_match) / float(len(output))) * 100


# returns sum of squared errors.
def calculate_erms(output, target):
    error = np.subtract(output,target)
    squared_error = np.square(error)
    sum_squared_error = np.sum(squared_error, axis=0)
    return math.sqrt(sum_squared_error/len(output))


def get_formatted_accuracy_and_error(accuracy, erms):
    return "Accuracy is = " + str(accuracy) + ", ERMS is = " + str(erms)


def closed_form_with_multiple_clusters():
    for M_value in M_array:
        global M
        M = M_value
        closed_form_solution()
    plot_clusters_vs_erms()
    plot_clusters_vs_accuracy()


def closed_form_with_multiple_lambda():
    for lamda_value in lambda_array:
        global lambda_constant
        lambda_constant = lamda_value
        closed_form_solution()
    plot_lambda_vs_erms()
    plot_lambda_vs_accuracy()


def plot_clusters_vs_erms():
    plt.figure()
    plt.plot(M_array, Training_Erms_Array, label="training")
    plt.plot(M_array, Validation_Erms_Array, label="validation")
    plt.plot(M_array, Testing_Erms_Array, label="testing")
    plt.legend()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Erms')
    plt.title('Clusters Vs ERMS')
    plt.show()


def plot_clusters_vs_accuracy():
    plt.figure()
    plt.plot(M_array, Training_Accuracy_Array, label="training")
    plt.plot(M_array, Validation_Accuracy_Array, label="validation")
    plt.plot(M_array, Testing_Accuracy_Array, label="testing")
    plt.legend()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Accuracy')
    plt.title('Clusters Vs Accuracy')
    plt.show()


def plot_lambda_vs_erms():
    plt.figure()
    plt.plot(lambda_array, Training_Erms_Array, label="training")
    plt.plot(lambda_array, Validation_Erms_Array, label="validation")
    plt.plot(lambda_array, Testing_Erms_Array, label="testing")
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('Erms')
    plt.title('Lambda Vs ERMS')
    plt.show()


def plot_lambda_vs_accuracy():
    plt.figure()
    plt.plot(lambda_array, Training_Accuracy_Array, label="training")
    plt.plot(lambda_array, Validation_Accuracy_Array, label="validation")
    plt.plot(lambda_array, Testing_Accuracy_Array, label="testing")
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('Erms')
    plt.title('Lambda Vs ERMS')
    plt.show()


def closed_form_solution():
    global training_phi_matrix, W, validation_phi_matrix, testing_phi_matrix
    clusters = KMeans(M, random_state=0).fit(trainingData)
    print("Created Clusters.")

    Means = clusters.cluster_centers_
    print("Calculated Means.")

    training_phi_matrix = calculate_phi_matrix(trainingData, BigSigma, Means)
    print("Calculated PHI for training data.")

    W = calculate_W_matrix(phi_matrix=training_phi_matrix, lam=lambda_constant)
    print("Calculated W matrix.")

    validation_phi_matrix = calculate_phi_matrix(validationData, BigSigma, Means)
    print("Calculated PHI for validation data.")

    testing_phi_matrix = calculate_phi_matrix(testingData, BigSigma, Means)
    print("Calculated PHI for testing data.")

    training_output = calculate_output(training_phi_matrix, W)
    validation_output = calculate_output(validation_phi_matrix, W)
    testing_output = calculate_output(testing_phi_matrix, W)
    print("Calculated the model output values")

    training_accuracy = calculate_accuracy(training_output, training_target)
    validation_accuracy = calculate_accuracy(validation_output, validation_target)
    testing_accuracy = calculate_accuracy(testing_output, testing_target)
    print("Calculated the Accuracy values")

    training_error = calculate_erms(training_output, training_target)
    validation_error = calculate_erms(validation_output, validation_target)
    testing_error = calculate_erms(testing_output, testing_target)
    print("Calculated the ERMS values")

    # Append the Erms to the array for plotting the graph
    Training_Erms_Array.append(training_error)
    Validation_Erms_Array.append(validation_error)
    Testing_Erms_Array.append(testing_error)

    # Append the Accuracy to the array for plotting the graph
    Training_Accuracy_Array.append(training_accuracy)
    Validation_Accuracy_Array.append(validation_accuracy)
    Testing_Accuracy_Array.append(testing_accuracy)

    print("Training Results: " + get_formatted_accuracy_and_error(training_accuracy, training_error))
    print("Validation Results: " + get_formatted_accuracy_and_error(validation_accuracy, validation_error))
    print("Testing Results: " + get_formatted_accuracy_and_error(testing_accuracy, testing_error))


def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) # * 100 is done to get the accuracy in percentage
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))



# Gradient Descent solution
def stochastic_gradient_descent_solution(learning_rate):
    La = 2
    current_weight = np.dot(220, W).flatten()
    L_Erms_Val = []
    L_Erms_TR = []
    L_Erms_Test = []
    L_Acc_TR = []
    L_Acc_Val = []
    L_Acc_Test = []
    for i in range(400):  # Number of data points on which training has been done
        b = (training_target[i] - np.dot(np.transpose(current_weight), training_phi_matrix[i]))
        delta_W = -np.multiply(b, training_phi_matrix[i])
        delta_W_x_La = np.dot(La, current_weight)
        delta_E = np.add(delta_W, delta_W_x_La)
        updated_W = -np.dot(learning_rate,delta_E)
        new_w = current_weight + updated_W
        current_weight = new_w # update the weight for next iteration

        # -----------------TrainingData Accuracy---------------------#
        # The output is calculated using y = w.transpose() * phi(x)
        TR_TEST_OUT = calculate_output(training_phi_matrix, new_w)
        Erms_TR = GetErms(TR_TEST_OUT, training_target.flatten())
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        L_Acc_TR.append(float(Erms_TR.split(',')[0]))

        # -----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT = calculate_output(validation_phi_matrix, new_w)
        Erms_Val = GetErms(VAL_TEST_OUT, validation_target.flatten())
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        L_Acc_Val.append(float(Erms_Val.split(',')[0]))

        # -----------------TestingData Accuracy---------------------#
        TEST_OUT = calculate_output(testing_phi_matrix, new_w)
        Erms_Test = GetErms(TEST_OUT, testing_target.flatten())
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        L_Acc_Test.append(float(Erms_Test.split(',')[0]))


    print("E_rms Training   = " + str(np.around(min(L_Erms_TR), 5)))
    print("E_rms Validation = " + str(np.around(min(L_Erms_Val), 5)))
    print("E_rms Testing    = " + str(np.around(min(L_Erms_Test), 5)))
    Training_Erms_Array.append(np.around(min(L_Erms_TR), 5))
    Validation_Erms_Array.append(np.around(min(L_Erms_Val), 5))
    Testing_Erms_Array.append(np.around(min(L_Erms_Test),5))
    Training_Accuracy_Array.append(np.around(max(L_Acc_TR), 5))
    Validation_Accuracy_Array.append(np.around(max(L_Acc_Val), 5))
    Testing_Accuracy_Array.append(np.around(max(L_Acc_Test), 5))


def sgd_with_multiple_learning_rate():
    learning_rate_array = [0.01, 0.05, 0.1, 0.3]
    for ele in learning_rate_array:
        stochastic_gradient_descent_solution(ele)



rawData = read_data_from_file(dataFile, True)
print(rawData.shape)

targetData = read_data_from_file(targetFile, False)
print(targetData.shape)

trainingData = get_sliced_data(rawData, training_percent, False, False)
training_target = get_sliced_data(targetData, training_percent, False, False)
print(trainingData.shape)

validationData = get_sliced_data(rawData, validation_percent, True, False)
validation_target = get_sliced_data(targetData, validation_percent, True, False)
print(validationData.shape)

testingData = get_sliced_data(rawData, testing_percent, False, True)
testing_target = get_sliced_data(targetData, testing_percent, False, True)
print(testingData.shape)

# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]
# Psudo inverse of a matrix will be helpful to give inverse even if it is a singular matrix (determinant(matrix) = 0)

BigSigma = get_big_sigma_matrix(rawData)

print ('UBITname      = kprabhak')
print ('Person Number = 50287403')
print ('----------------------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
closed_form_solution()
print ('----------Gradient Descent Solution--------------------')
print ('----------------------------------------------------')
stochastic_gradient_descent_solution(0.01)

#closed_form_with_multiple_clusters()
#closed_form_with_multiple_lambda()
#sgd_with_multiple_learning_rate()



