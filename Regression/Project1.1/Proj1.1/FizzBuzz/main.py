from Software1 import *
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard


def createInputCSV(start, end, filename):
    # Why list in Python?
    # Lists are basic collections in python. Lists are mutable.
    # No restriction in size and increases dynamically based on the data.
    # Also, unlike C,C++, as there are no arrays in Python.
    # In our case, we can append any number of training and testing data sets,
    # which are later copied to columns in our CSV
    # Most of the libraries(numpy,pandas and matplotlib) extensively uses data in form of lists(Data set example below).
    # It can also be indexed. It also allows negative index to retrieve the elements from the end of the list

    inputData = []
    outputData = []

    # Why do we need training Data?
    # As we are following supervised learning approach,
    # we build our model based on the right results available from the training data
    # The training data is a kind of past experience(E) for the system to predict the model that suits for future inputs

    for i in range(start, end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # Why Dataframe?
    # Pandas is a well known library, which has data structures to collect and process data in the form tables.
    # It has lot of functionality available to work on table data which we use here for constructing our CSV.
    # The data frame method from this library uses the data set dictionary and converts it to the csv file.
    dataset = {}
    dataset["input"] = inputData
    dataset["label"] = outputData

    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")


def processData(dataset):
    # Why do we have to process?
    # The data we have in our training and testing set is in decimal form. When we give this input to our neural
    # network, we need to convert it to binary form so that each node gets a particular bit from the number.
    # For example, if the input is 3, it is processed as [1,1,0,0,0,0,0,0,0,0] where the bits are given to each node
    # [x0,x1,x2,x3,x4,x5,x6x,x7,x8,x9]
    data = dataset['input'].values
    labels = dataset['label'].values

    processedData = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


def encodeData(data):
    processedData = []

    for dataInstance in data:
        # Why do we have number 10?
        # we need to convert the decimal to 10 bit binary (since the maximum size of our data set is 1024(2^10))
        # The below operation  right shifts each number for 1 to 10 times and then does AND operation. This will give
        # proper representation of the input data for the network.
        # We can still develop our model with out reversing the bits by just converting each input into 10 bit binary.
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)


def encodeLabel(labels):
    processedLabel = []

    for labelInstance in labels:
        if (labelInstance == "fizzbuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif (labelInstance == "fizz"):
            # Fizz
            processedLabel.append([1])
        elif (labelInstance == "buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel), 4)


input_size = 10
drop_out = 0.1
first_dense_layer_nodes = 512
second_dense_layer_nodes = 512
final_layer_nodes = 4
def get_model():
    # Why do we need a model?
    # A model is a mathematical function used to represent the training data. We use model to make prediction on future
    # data. It is the functional notation of acquired experience from the training set.

    # Why use Dense layer and then activation?
    # The activation is applied on the result of matrix multiplication of weight matrix and input vector.
    # To perform this multiplication, we first need to set the weights of the nodes and input values.
    # Hence, we first define the dense layers with acceptable values and then we introduce the activation function.

    # Why use sequential model with layers?
    # We are dealing with a classification problem, where we need a kind of transformation function, which can
    # effectively reduce the different outcomes and provides the true outcome with high probability.
    # As the model passes through different layers, the sequential model restricts the irrelevant outcomes and scopes
    # down the relevant outputs to a small set.
    # This operation is repeatedly performed and provides proper output in the final layer
    # Using the functional API's in Keras, where we can connect a layer to other layer not just previous and next layer.
    # But for the simple classification problems, will be much easier to use.

    model = Sequential()

    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    # The activation function provides non linearity. As it is evident that a linear model cant cover all the
    # data points and also will not be appropriate for different testing data sets.
    # So we need to introduce some non-linearity into our model for best prediction of feature inputs. It is provided by
    # activation functions.
    model.add(Activation('relu'))

    # Why dropout?
    # To understand drop out, we need to understand over-fitting, which means our model is too exact to particular
    # training set and will fail to reliably predict the output for future inputs.
    # To avoid this we introduce the concept called dropout, where in neural network, we reduce the number of neurons by
    # the dropout factor. For Eg: If we initially use 100 neurons and we end up with over-fit model, we now introduce a
    # dropout of 0.2 which will reduce the neurons to 80 (100*0.2) to get the appropriate model

    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes, input_dim=first_dense_layer_nodes))
    model.add(Activation('relu'))

    model.add(Dense(final_layer_nodes))
    model.add(Activation('softmax'))

    # Why Softmax?
    # The output of a softmax function represents the categorical distribution where the arbitrary values are reduced to
    # to fit in to the range(0,1). As we are dealing with  a classification problem,
    # its ideal to use a softmax to classify the output in (0,1) range, which is later decoded to required result.

    model.summary()

    # Why use categorical_crossentropy?
    # The compile method is used to for configuring the learning process by using optimizer, loss function and metrics
    # As this is a multi class output problem and also our targets are in one-hot encoding, which means representation
    # of output variables as binary vectors like [1,0,0,0] - Fizz, etc., we use categorical_crossentropy
    # Also the use of categorical_crossentropy is recommended because, we have earlier categorised our target labels
    # using to_categorical method, which is recommended way to use categorical_crossentropy
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "other"
    elif encodedLabel == 1:
        return "fizz"
    elif encodedLabel == 2:
        return "buzz"
    elif encodedLabel == 3:
        return "fizzbuzz"


createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')
model = get_model()


validation_data_split = 0.2 # 0.2 times of the validation data is used for validation and the rest 0.8 for training
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

# Tensorboard provides the features for visualizing the learning of our model
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
# Over fitting is a case where the model becomes very specific to the input data set and couldn't output
# the expected results for all inputs outside the training data set.
# We perform early stopping to avoid over fitting problem
# Also early stopping will stop training when a monitored quantity has stopped growing.
# Patience is number of epochs with no improvement after which training will be stopped.
# In mode (is trajectory of the quantity): min means decreasing. max means increasing.
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb, earlystopping_cb]
                   )


df = pd.DataFrame(history.history)
import matplotlib.pyplot as plt

df.plot(subplots=True, grid=True, figsize=(10,15))
plt.show()

wrong = 0
right = 0

testData = pd.read_csv('testing.csv')

processedTestData = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)

predictedTestLabel = []

for i, j in zip(processedTestData, processedTestLabel):
    temp = np.array(i).reshape(-1,10)
    y = model.predict(temp)
    predictedTestLabel.append(decodeLabel(y.argmax()))

    if j.argmax() == y.argmax():
        right = right + 1 # If there is a match between training and testing sets, then right is incremented
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right / (right + wrong) * 100))

testDataInput = testData['input'].tolist();
testDataLabel = testData['label'].tolist();
testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "Kprabhak")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50287403")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel
output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')