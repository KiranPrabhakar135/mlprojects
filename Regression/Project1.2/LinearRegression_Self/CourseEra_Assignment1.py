import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
M_array = [0.01, 0.05, 0.1]
Training_Erms_Array = [74.52153, 74.52153, 74.52153]
Validation_Erms_Array = [75.25136, 75.19391, 75.17955]
Testing_Erms_Array = [70.29589, 70.23844, 70.23844]
def clusters_vs_erms():
    plt.figure()
    plt.plot(M_array, Training_Erms_Array, label = "training")
    #plt.legend()
    plt.plot(M_array, Validation_Erms_Array, label = "validation")
    #plt.legend("validation")
    plt.plot(M_array, Testing_Erms_Array, label = "testing")
    plt.legend()
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.title('Learning rate Vs Accuracy')
    plt.show()

    # plt.figure(figsize=[6, 6])
    # plt.plot(M_array, Validation_Erms_Array)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Validation Erms')
    # plt.title('Clusters Vs ERMS')
    # plt.show()
    # plt.figure(figsize=[6, 6])
    # plt.plot(M_array, Testing_Erms_Array)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Testing Erms')
    # plt.title('Clusters Vs ERMS')
    # plt.show()

clusters_vs_erms()
filepath = "C:\ML\CourseEra\machine-learning-ex1\ex1\Sample_LinearRegression.csv"

def GetData(filePath):
    data = pd.read_csv(filePath, sep="\n");
    return data

def ScatterPlotData(data):
    for dataitem in data.__array__():
        dataitem1 = []
        dataitem1.append(float(dataitem[0].split(',')[0].strip()))
        dataitem1.append(float(dataitem[0].split(',')[1].strip()))
        data.append(dataitem1)
    df = pd.DataFrame(data, columns=["population", "profit"])
    df.plot.scatter(x="population", y="profit")
    return  data

def getCoulmnValues(column, append1ToInput):
    X = []
    data = GetData(filepath)
    for item in data:
        temp = []
        if(append1ToInput):
            temp.append(1);
        temp.append(item[column])
        X.append(temp)
    print(X)
    return X

def PerformRegression():
    X = getCoulmnValues(0, True)
    rawInputs = getCoulmnValues(0, False)
    X = np.array(X);
    Y = np.array(getCoulmnValues(1, False))
    W = np.zeros([1, 2]);
    m = 30
    alpha = 0.001
    print(W.shape)
    for i in range(0, 9000):
        H = np.matmul(X, W.transpose())
        Cost = np.subtract(H, Y)
        Sum0 = np.sum(Cost)

        CostArray = np.array(Cost);
        Cost = np.array(np.multiply(CostArray, rawInputs));

        Sum1 = np.sum(Cost)
        print(W)
        print(Sum0)
        print(Sum1)
        W[0][0] = W[0][0] - (Sum0 / m) * alpha

        W[0][1] = W[0][1] - (Sum1 / m) * alpha
    return W

#ScatterPlotData(GetData(filepath))
#PerformRegression()

