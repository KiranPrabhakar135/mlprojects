import pickle as pkl
from PIL import Image
import os
import numpy as np
import pandas as pd
import math
import seaborn as sb
import matplotlib.pyplot as plt

def get_usps_data_from_images():
    USPSMat_arr = []
    USPSTar_arr = []
    curPath = 'Data/USPSdata/Numerals'
    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat_arr.append(imgdata)
                USPSTar_arr.append(j)
    return USPSMat_arr, USPSTar_arr


def get_mnist_data():
    with open("Data/mnist.pkl", 'rb') as pkl_file:
        training_data, validation_data, test_data = pkl.load(pkl_file, encoding='latin1')
        return training_data,validation_data,test_data


def get_usps_data():
    with open("Data/usps.pkl", 'rb') as pkl_file:
        testing_data, test_target = pkl.load(pkl_file, encoding='latin1')
        return np.array(testing_data), np.array(test_target)


def concatenate_mnist_and_usps_testing_data(mnist_data, usps_data):
    return np.append(mnist_data,usps_data, axis=0)


def calculate_accuracy_and_error(output, testing_target):
    diff_bwn_output_target = np.subtract(output, testing_target)
    output_length = diff_bwn_output_target.shape[0]
    error_value = np.count_nonzero(diff_bwn_output_target) / output_length
    accuracy_value = 1 - error_value
    return accuracy_value*100, error_value*100


def calculate_confusion_matrix(output, testing_target):
    confusion_matrix_py = np.zeros((10,10), dtype="int")
    for i, j in zip(output, testing_target):
        confusion_matrix_py[i][j] += 1
    print(confusion_matrix_py)
    return confusion_matrix_py


def create_heat_map(confusion_matrix):
    dataframe = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10,10))
    sb.heatmap(dataframe, annot=True, fmt="g", cbar=False)
    plt.show()


mnist_total_training_data, mnist_total_validation_data, mnist_total_testing_data = get_mnist_data()
mnist_training_data = mnist_total_training_data[0]
mnist_training_target = mnist_total_training_data[1]
mnist_one_hot_vector_training_target = np.array(np.eye(10)[mnist_training_target])
mnist_validation_data = mnist_total_validation_data[0]
mnist_validation_target = mnist_total_validation_data[1]
mnist_testing_data = mnist_total_testing_data[0]
mnist_testing_target = mnist_total_testing_data[1]
mnist_one_hot_vector_testing_target = np.array(np.eye(10)[mnist_testing_target])

usps_testing_data, usps_testing_target = get_usps_data()

mnist_and_usps_testing_data = concatenate_mnist_and_usps_testing_data(mnist_testing_data, usps_testing_data)
mnist_and_usps_testing_target = concatenate_mnist_and_usps_testing_data(mnist_testing_target, usps_testing_target)

mnist_appended_training_data_and_target = np.concatenate((np.array(mnist_training_data), np.array(mnist_training_target).reshape(-1,1)), axis=1)


def get_bag(data):
    data = np.array(data)
    np.random.shuffle(data)
    total_data_items = data.shape[0]
    sliced_data = data[0:math.ceil(total_data_items*0.6), :]
    bagged_data = sliced_data[:,:-1]
    bagged_target = sliced_data[:, -1]
    return bagged_data, bagged_target