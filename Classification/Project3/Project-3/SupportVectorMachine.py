import DataProcessing as data
from sklearn import svm


def generate_model(kernal, regularizer, gamma, training_data, training_target):
    svm_classifier = svm.SVC(kernel=kernal , C=regularizer, gamma=gamma)
    svm_classifier.fit(training_data, training_target)
    return svm_classifier


def predict_output(svm_classifier, testing_data):
    return svm_classifier.predict(testing_data)


def perform_classification_and_output_accuracy(model):
    model_output = predict_output(model, data.mnist_testing_data).astype("int")
    accuracy, error = data.calculate_accuracy_and_error(model_output, data.mnist_testing_target)
    print("SVM: Mnist testing: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(model_output, data.mnist_testing_target)
    data.create_heat_map(confusion_matrix)
    model_output = predict_output(model, data.usps_testing_data).astype("int")
    accuracy, error = data.calculate_accuracy_and_error(model_output, data.usps_testing_target)
    print("SVM: Usps testing: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(model_output, data.usps_testing_target)
    data.create_heat_map(confusion_matrix)


def create_model_perform_classification_and_output_accuracy():
    model = generate_model("rbf", 1, 0.01, data.mnist_training_data, data.mnist_training_target)
    perform_classification_and_output_accuracy(model)

#create_model_perform_classification_and_output_accuracy()