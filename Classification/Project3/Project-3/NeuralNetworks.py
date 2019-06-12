import numpy as np
import DataProcessing as data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation


def create_model(input_size, final_layer_nodes):
    first_dense_layer_nodes = 512
    second_dense_layer_nodes = 512
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))

    model.add(Activation('relu'))

    model.add(Dense(second_dense_layer_nodes, input_dim= first_dense_layer_nodes))

    model.add(Activation('relu'))

    model.add(Dense(final_layer_nodes, input_dim=second_dense_layer_nodes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def fit_model_to_data(model, training_data, target, epochs):
    target = np_utils.to_categorical(np.array(target), 10)
    model.fit(training_data, target, batch_size=128, epochs=epochs, verbose=False)
    return model


def predict_output(model, test_data):
    result = []
    for item in test_data:
        temp = np.array(item).reshape(-1, test_data.shape[1])
        result.append(model.predict(temp).argmax())
    return np.array(result)


def perform_classification_and_output_accuracy(nn_model):
    model_output = predict_output(nn_model, data.mnist_testing_data).astype("int")
    accuracy, error = data.calculate_accuracy_and_error(model_output, data.mnist_testing_target)
    print("Neural Networks: Mnist testing: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(model_output, data.mnist_testing_target)
    data.create_heat_map(confusion_matrix)
    model_output = predict_output(nn_model, data.usps_testing_data).astype("int")
    accuracy, error = data.calculate_accuracy_and_error(model_output, data.usps_testing_target)
    print("Neural Networks: Usps testing: " + str(accuracy))
    confusion_matrix = data.calculate_confusion_matrix(model_output, data.usps_testing_target)
    data.create_heat_map(confusion_matrix)


def create_model_perform_classification_and_output_accuracy():
    nn_model = create_model(784, 10)
    nn_model = fit_model_to_data(nn_model, data.mnist_training_data, data.mnist_training_target, 100)
    perform_classification_and_output_accuracy(nn_model)

#create_model_perform_classification_and_output_accuracy()