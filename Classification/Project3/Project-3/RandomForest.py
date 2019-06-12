import numpy as np
import DataProcessing as data
from sklearn.ensemble import RandomForestClassifier


def generate_model(trees_count, training_data, training_target):
    random_forest_classifier = RandomForestClassifier(n_estimators=trees_count)
    random_forest_classifier.fit(training_data, training_target)
    return random_forest_classifier


def predict_output(random_forest_model, testing_data):
    return random_forest_model.predict(testing_data)


def perform_classification_and_output_accuracy(model):
    model_output = predict_output(model, data.mnist_testing_data).astype("int")
    accuracy, error = data.calculate_accuracy_and_error(model_output, data.mnist_testing_target)
    print("RandomForest: Mnist testing: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(model_output, data.mnist_testing_target)
    data.create_heat_map(confusion_matrix)
    model_output = predict_output(model, data.usps_testing_data).astype("int")
    accuracy, error = data.calculate_accuracy_and_error(model_output, data.usps_testing_target)
    print("RandomForest: Usps testing: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(model_output, data.usps_testing_target)
    data.create_heat_map(confusion_matrix)


def create_model_perform_classification_and_output_accuracy():
    model = generate_model(100, data.mnist_training_data, data.mnist_training_target)
    perform_classification_and_output_accuracy(model)


#create_model_perform_classification_and_output_accuracy()