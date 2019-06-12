import pandas as pd
import numpy as np
import math


# for index, row in data.iterrows():
#     first_image = row['img_id_A']
#     first_image_feature_details = human_or_gsc_data.loc[human_or_gsc_data['img_id'] == first_image].values
#     if is_gsc_data:
#         first_image_features = first_image_feature_details[0][1:]
#     else:
#         first_image_features = first_image_feature_details[0][2:]
#
#     second_image = row["img_id_B"]
#     second_image_feature_details = human_or_gsc_data.loc[human_or_gsc_data['img_id'] == second_image].values
#     if is_gsc_data:
#         second_image_features = second_image_feature_details[0][1:]
#     else:
#         second_image_features = second_image_feature_details[0][2:]
#
#     concatenation_result = np.concatenate((first_image_features, second_image_features))
#     subtraction_result = np.absolute(first_image_features - second_image_features)
#     concatenated_array.append(concatenation_result)
#     subtracted_array.append(subtraction_result)

def h(theta,X): #Linear hypothesis function
    return np.dot(X,theta)


def descendGradient(X, theta_start = np.zeros(2)):
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    """
    theta = theta_start
    jvec = [] #Used to plot cost as function of iteration
    thetahistory = [] #Used to visualize the minimization path later on
    for meaninglessvariable in range(100):
        tmptheta = theta
        #jvec.append(computeCost(theta,X,y))
        # Buggy line
        #thetahistory.append(list(tmptheta))
        # Fixed line
        thetahistory.append(list(theta[:,0]))
        #Simultaneously updating theta values
        for j in range(len(tmptheta)):
            d = np.sum((h(initial_theta,X) - y)*np.array(X[:,j]).reshape(m,1));
            tmptheta[j] = theta[j] - (0.0001/m)*np.sum((h(initial_theta,X) - y)*np.array(X[:,j]).reshape(m,1))
        theta = tmptheta
        print(theta)
    return theta, thetahistory, jvec

cols = np.loadtxt(r'C:\ML\CourseEra\SingleVariableLinearRegression.txt',delimiter=',',usecols=(0,1),unpack=True) #Read in comma separated data
#Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size # number of training examples
#Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)
initial_theta = np.zeros((X.shape[1],1))
#descendGradient(X, initial_theta)



data = pd.read_csv(r'C:\ML\CourseEra\SingleVariableLinearRegression.csv')
total_data = data.loc[:, 'x0':'x1']
total_target_data = data.loc[:, 't':'t']
total_data_count = total_data.shape[0]
training_data_end_index = math.ceil(total_data_count*0.8)
validation_data_end_index = math.ceil(total_data_count*0.9)
training_data = total_data.head(training_data_end_index)
validation_data = total_data.loc[training_data_end_index+1:validation_data_end_index,:]
testing_data = total_data.loc[validation_data_end_index+1:,:]
training_target = total_target_data.head(training_data_end_index)
validation_target = total_target_data.loc[training_data_end_index+1:validation_data_end_index,:]
testing_target = total_target_data.loc[validation_data_end_index+1:,:]
weight_matrix = np.zeros((total_data.shape[1], 1))
learning_rate = 0.0001

updated_weights = np.zeros((total_data.shape[1], 1))
training_data_length = training_data.shape[0]

for epoch in range(1,11):
    #updated_weights = np.subtract(weight_matrix,(learning_rate/total_data_count) * (np.dot(total_data,weight_matrix) - np.array(total_target_data)).transpose() * np.array(total_data).transpose())

    for index_n in range(weight_matrix.shape[0]):
        derivative = np.sum((np.dot(total_data, weight_matrix)- np.array(total_target_data))* np.array(total_data.iloc[:, index_n: index_n + 1]))
        # for index_m, row_m in total_data.iterrows():
        #     y = np.dot(weight_matrix.transpose(),np.array(row_m).reshape((-1,1)))
        #     derivative = derivative + ((y - total_target_data.iloc[index_m][0])*row_m[index_n])[0][0]
        updated_weights[index_n] = weight_matrix[index_n][0] - learning_rate*(derivative/total_data_count)

    weight_matrix = updated_weights
    print(weight_matrix)


print("Final weight matrix is: ")
print(weight_matrix)
# print(total_data)
# print(training_data)
# print(testing_target)