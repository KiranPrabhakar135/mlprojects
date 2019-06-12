import DataProcessing as data
import LogisticRegression as log_reg
import SupportVectorMachine as svm
import NeuralNetworks as nn
import RandomForest as rf
import numpy as np
import statistics as stat


def perform_logistic_regression():
    training_data, training_target = data.get_bag(data.mnist_appended_training_data_and_target)
    return log_reg.sigmoidal_regression(1000, 0.5, training_data, np.array(np.eye(10)[training_target.astype(int)]))


def perform_svm_classification(kernal):
    training_data, training_target = data.get_bag(data.mnist_appended_training_data_and_target)
    return svm.generate_model(kernal, 100, 0.01, training_data, training_target)


def perform_nn_classification():
    training_data, training_target = data.get_bag(data.mnist_appended_training_data_and_target)
    model = nn.create_model(784, 10)
    model = nn.fit_model_to_data(model, training_data, training_target, 100)
    return model


def perform_rf_classification():
    training_data, training_target = data.get_bag(data.mnist_appended_training_data_and_target)
    return rf.generate_model(100, training_data, training_target)


def perform_max_voting_classification():
    print("....Creating all Classifiers using bagging for Max Voting....")
    logistic_weights = perform_logistic_regression()
    print("....Completed Softmax regression....")
    svm_model_linear = perform_svm_classification("linear")
    print("....Completed SVM with linear kernal....")
    svm_model_rbf = perform_svm_classification("rbf")
    print("....Completed SVM with rbf kernal....")
    nn_model = perform_nn_classification()
    print("....Completed Neural network....")
    random_forest_model = perform_rf_classification()
    print("....Completed Random forests....")

    print("....Performing Max Voting classifications....")
    perform_max_voting(logistic_weights, svm_model_linear, svm_model_rbf,nn_model, random_forest_model)


def perform_max_voting(logistic_weights, svm_model_linear, svm_model_rbf, nn_model, random_forest_model):
    accuracy, error = calculate_accuracy_using_max_vote(data.mnist_testing_data, data.mnist_testing_target, logistic_weights, nn_model,
                                                        random_forest_model, svm_model_linear, svm_model_rbf)
    print("Ensemble Classifier: Mnist testing: " + str(accuracy))

    accuracy, error = calculate_accuracy_using_max_vote(data.usps_testing_data, data.usps_testing_target, logistic_weights,
                                                        nn_model, random_forest_model, svm_model_linear, svm_model_rbf)
    print("Ensemble Classifier: USPS testing: " + str(accuracy))


def calculate_accuracy_using_max_vote(testing_data, testing_target, logistic_weights, nn_model, random_forest_model, svm_model_linear, svm_model_rbf):
    ensemble_output = []
    i = -1
    for row in testing_data:
        i += 1
        temp = []
        row = np.array(row).reshape(1, -1)
        temp.append(np.argmax(log_reg.calculate_output(logistic_weights, row)))
        temp.append(int(svm.predict_output(svm_model_linear, row)[0]))
        temp.append(int(svm.predict_output(svm_model_rbf, row)[0]))
        temp.append(int(nn.predict_output(nn_model, row)[0]))
        temp.append(int(rf.predict_output(random_forest_model, row)[0]))
        try:
            ensemble_output.append(stat.mode(temp))
        except:
            temp_error = []
            for ele in temp:
                actual_target = testing_target[i]
                error = (ele - actual_target)
                if error == 0:
                    ensemble_output.append(ele)
                    temp_error.clear()
                    break
                else:
                    temp_error.append(abs(error))
            if len(temp_error) > 0:
                ensemble_output.append(temp[np.argmin(temp_error)])
                temp_error.clear()
    confusion_matrix = data.calculate_confusion_matrix(ensemble_output, testing_target)
    data.create_heat_map(confusion_matrix)
    return data.calculate_accuracy_and_error(ensemble_output, testing_target)


#perform_max_voting_classification()