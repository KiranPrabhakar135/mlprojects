from Software1 import *

# coding: utf-8

# In[54]:


import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
import ipywidgets as widgets
import matplotlib.pyplot as plt





# ## Logic Based FizzBuzz Function [Software 1.0]

# In[55]:


def fizzbuzz(n):
    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[56]:


def createInputCSV(start, end, filename):
    # Why list in Python?
    inputData = []
    outputData = []

    # Why do we need training Data?
    for i in range(start, end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # Why Dataframe?
    dataset = {}
    dataset["input"] = inputData
    dataset["label"] = outputData

    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")


# ## Processing Input and Label Data

# In[57]:


def processData(dataset):
    # Why do we have to process?
    data = dataset['input'].values
    labels = dataset['label'].values

    processedData = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


# In[58]:


import numpy as np


def encodeData(data):
    processedData = []

    for dataInstance in data:
        # Why do we have number 10?
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)


# In[59]:


def encodeLabel(labels):
    processedLabel = []

    for labelInstance in labels:
        if (labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif (labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif (labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel), 4)


# In[60]:


# Create datafiles
createInputCSV(101, 1001, 'training.csv')
createInputCSV(1, 101, 'testing.csv')

# In[61]:


# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel = processData(testingData)

# ## Tensorflow Model Definition

# In[62]:


# Defining Placeholder
inputTensor = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])

# In[66]:


NUM_HIDDEN_NEURONS_LAYER_1 = 100
LEARNING_RATE = 0.05


# Initializing the weights to Normal Distribution
# the function will return variable which has shape(10,100) if the shape input is [10,100] and
# the shape(10, 100) implies there is an array of 10 arrays where each array has 100 elements.
def init_weights(shape):
   return tf.Variable(tf.random_normal(shape, stddev=0.01));


# Initializing the input to hidden layer weights
input_hidden_weights = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])
# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])

# Computing values at the hidden layer
# the weight matrix is multiplied with the input and then the result is given to the activation function to get the
# hidden layer output.
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))

# Computing values at the output layer
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
# Logits are unnormalized log probability, which are usually given as input to the softmax function, which will normalize
# the sum to 1.
# Labels will give the shape you are expecting from the function after the softmax operation on the logits.
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
# once the gradient descent is optimized, it gives the training model.
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
prediction = tf.argmax(output_layer, 1)

# # Training the Model

# In[67]:


NUM_OF_EPOCHS = 5000
BATCH_SIZE = 128

training_accuracy = []

with tf.Session() as sess:
    # Set Global Variables ?
    tf.global_variables_initializer().run()

    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):

        # Shuffle the Training Dataset at each epoch
        p = np.random.permutation(range(len(processedTrainingData)))
        processedTrainingData = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]

        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end],
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                                         sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                                         outputTensor: processedTrainingLabel})))
    # Testing
    test_accuracy = []
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData});
    test_accuracy .append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==  predictedTestLabel));
    print(test_accuracy)
# In[68]:


df = pd.DataFrame()
df['acc'] = training_accuracy
#df['loss'] = test_accuracy
df.plot(grid=True)
plt.show()


# In[69]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# # Testing the Model [Software 2.0]

# In[71]:


wrong = 0
right = 0

predictedTestLabelList = []
""
for i, j in zip(processedTestingLabel, predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))

    if np.argmax(i) == j:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right / (right + wrong) * 100))

# Please input your UBID and personNumber
testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "Kprabhak")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50287403")

predictedTestLabelList.insert(0, "")
predictedTestLabelList.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabelList

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

