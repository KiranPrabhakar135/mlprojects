import numpy as np
import pandas as pd
from keras.utils import np_utils
import DataSet.DataProcessing as data
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard


def get_model(input_size):
    drop_out = 0.1
    first_dense_layer_nodes = 512
    second_dense_layer_nodes = 512
    final_layer_nodes = 2

    model = Sequential()

    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))

    model.add(Activation('sigmoid'))

    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes, input_dim=first_dense_layer_nodes))
    model.add(Activation('sigmoid'))

    model.add(Dense(final_layer_nodes))
    model.add(Activation('softmax'))

    #model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def process_model(number_of_inputs,training_data, training_target):
    validation_data_split = 0.25
    num_epochs = 10000
    model_batch_size = 128
    tb_batch_size = 32
    early_patience = 100
    model = get_model(number_of_inputs)
    tensorboard_cb = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    training_target = np_utils.to_categorical(np.array(training_target), 2)

    training_result = model.fit(training_data
                        , training_target
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb, earlystopping_cb]
                        , verbose = 0
                       )

    return training_result, model


def validate_model(model, test_data, test_result):
    right = 0
    wrong = 0
    test_data = np.array(test_data)
    test_result = np.array(test_result)
    for i, j in zip(test_data, test_result):
        temp = np.array(i).reshape(-1, test_data.shape[1])
        y = model.predict(temp)

        if j.argmax() == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1

    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("Testing Accuracy: " + str(right / (right + wrong) * 100))


def perform_classification(training_data, training_target, testing_data, testing_target):
    input_size = training_data.shape[1]
    training_result, model = process_model(input_size, training_data, training_target)
    df = pd.DataFrame(training_result.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))
    plt.show()
    validate_model(model, testing_data, testing_target)


def con_hum():
    perform_classification(data.concatenated_human_training_data.append(data.concatenated_human_validation_data),
                       data.concatenated_human_training_target_data.append(data.concatenated_human_validation_target_data),
                       data.concatenated_human_testing_data, data.concatenated_human_testing_target_data)


def sub_hum():
    perform_classification(data.subtracted_human_training_data.append(data.subtracted_human_validation_data),
                           data.subtracted_human_training_target_data.append(
                               data.subtracted_human_validation_target_data),
                           data.subtracted_human_testing_data, data.subtracted_human_testing_target_data)


def con_gsc():
    perform_classification(data.concatenated_gsc_training_data.append(data.concatenated_gsc_validation_data),
                           data.concatenated_gsc_training_target_data.append(
                               data.concatenated_gsc_validation_target_data),
                           data.concatenated_gsc_testing_data, data.concatenated_gsc_testing_target_data)


def sub_gsc():
    perform_classification(data.subtracted_gsc_training_data.append(data.subtracted_gsc_validation_data),
                           data.subtracted_gsc_training_target_data.append(
                               data.subtracted_gsc_validation_target_data),
                           data.subtracted_gsc_testing_data, data.subtracted_gsc_testing_target_data)