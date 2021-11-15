import sys
import numpy as np
import pandas as pd
from scipy.special import expit
import time

def softmax(arr):
    'Function to find the softmax distribution'

    e = np.exp(arr - arr.max())
    return e / e.sum()

def derivativeSoftmax(arr):
    'Function to find the softmax derivative of an array'

    e = np.exp(arr - arr.max())
    return e / e.sum() * (1 - e / e.sum())

def derivativeSigmoid(arr):
    'Function to find the sigmoid derivative of an array'

    e = arr.applymap(expit)
    return e * (1 - e)

class DigitClassifier:

    def __init__(self, train_images_file, train_label_file, test_images_file):
        'Function to initialize the neural network'

        #Setting hyperparameters
        self.inputs = 784
        self.hidden_units_1 = 256
        # self.hidden_units_2 = 64
        self.outputs = 10
        self.epochs = 100
        self.batch_size = 1000
        self.learning_rate = 0.0025

        #Initializing Weights
        # self.weights_1 = pd.DataFrame(np.random.uniform(-2, 2, (self.inputs + 1, self.hidden_units_1)))         #+1 for bias
        # self.weights_2 = pd.DataFrame(np.random.uniform(-2, 2, (self.hidden_units_1, self.hidden_units_2)))
        # self.weights_out = pd.DataFrame(np.random.uniform(-2, 2, (self.hidden_units_2, self.outputs)))


        self.weights_1 = pd.DataFrame(np.random.uniform(-2, 2, (self.inputs + 1, self.hidden_units_1)) * np.sqrt(1/(self.inputs + 1 + self.hidden_units_1)))
        # self.weights_2 = pd.DataFrame(np.random.uniform(-2, 2, (self.hidden_units_1, self.hidden_units_2)) * np.sqrt(1/(self.hidden_units_1 + self.hidden_units_2)))
        self.weights_out = pd.DataFrame(np.random.uniform(-2, 2, (self.hidden_units_1, self.outputs)) * np.sqrt(1/(self.hidden_units_1 + self.outputs)))
        # self.weights_out = pd.DataFrame(np.random.uniform(-2, 2, (self.hidden_units_2, self.outputs)) * np.sqrt(1/(self.hidden_units_2 + self.outputs)))

        print('Neural Network Initialized')

        #Importing training data and labels
        with open(train_images_file) as f:
            self.train_data = pd.read_csv(f, header=None)

        with open(train_label_file) as f:
            labels = pd.read_csv(f, header=None)
            self.train_data['Label'] = labels

        self.train_data.columns = [i for i in range(self.inputs)] + ['Label']
        self.train_data.applymap(int)
        self.train_data = self.train_data[:10000]                                   #remove
        print('Training Data Imported')

        #Importing testing data
        with open(test_images_file) as f:
            self.test_data = pd.read_csv(f, header=None)

        self.test_data.columns = [i for i in range(self.inputs)]
        self.test_data.applymap(int)
        print('Testing Data Imported')

    def update_weights(self, update_output, update_1):
    # def update_weights(self, update_output, update_2, update_1):
        'Function to update weights'

        self.weights_out -= self.learning_rate * update_output
        # self.weights_2 -= self.learning_rate * update_2
        self.weights_1 -= self.learning_rate * update_1

    def backPropagate(self, labels, softmax_output, output_wx, activated_first_wx, first_wx, data):
    # def backPropagate(self, labels, softmax_output, output_wx, activated_second_wx, second_wx, activated_first_wx, first_wx, data):
        'Function to back propagate error and adjust weights'

        cross_entropy_loss = -labels * np.log(softmax_output)           #visualize or remove

        #Output Layer Error
        error_output = (softmax_output - labels) * derivativeSoftmax(output_wx)
        update_output = activated_first_wx.T.dot(error_output)
        # update_output = activated_second_wx.T.dot(error_output)

        # #Second Layer Error
        # error_2 = ((self.weights_out).dot(error_output.T)).T * derivativeSigmoid(second_wx)
        # update_2 = activated_first_wx.T.dot(error_2)

        #First Layer Error
        # error_1 = ((self.weights_2).dot(error_2.T)).T * derivativeSigmoid(first_wx)
        error_1 = ((self.weights_out).dot(error_output.T)).T * derivativeSigmoid(first_wx)
        update_1 = data.T.dot(error_1)

        self.update_weights(update_output, update_1)
        # self.update_weights(update_output, update_2, update_1)

    def feedForward(self, batch, predict = True):
        'Function to compute the output at each layer'

        data = batch.loc[:, batch.columns != 'Label']
        data[784] = [1 for _ in range(len(batch.index))]        #adding bias node to input layer

        if not predict:
            #One-hot encoding output
            labels = np.zeros((self.batch_size, 10))
            rows  = range(self.batch_size)
            labels[rows, batch['Label']] = 1

        #1st layer
        first_wx = data.dot(self.weights_1)                     #multiply by weights
        activated_first_wx = first_wx.applymap(expit)           #pass through activation function

        #Add bias to second and output layers?

        # #2nd layer
        # second_wx = activated_first_wx.dot(self.weights_2)      #multiply by weights
        # activated_second_wx = second_wx.applymap(expit)         #pass through activation function

        #Output layer
        output_wx = activated_first_wx.dot(self.weights_out)   #multiply by weights
        # output_wx = activated_second_wx.dot(self.weights_out)   #multiply by weights
        softmax_output = output_wx.apply(softmax, axis=1)       #pass through activation function

        final_output = softmax_output.idxmax(axis=1)

        if predict:
            return final_output
        else:
            accuracy = (final_output==batch['Label']).mean()
            # if accuracy == 0:
            #     print('Accuracy == 0 Error')
            #     exit()
            self.backPropagate(labels, softmax_output, output_wx, activated_first_wx, first_wx, data)
            # self.backPropagate(labels, softmax_output, output_wx, activated_second_wx, second_wx, activated_first_wx, first_wx, data)
            return accuracy

    def train(self):
        'Function to train the network using training data'

        shuffled_train_data = self.train_data.sample(frac=1)
        batches = np.array_split(shuffled_train_data, (self.train_data.shape[0] + 1) / self.batch_size)
        print('Training Data Segmented')

        print('Training Started')
        for epoch in range(self.epochs):
            for batch in range(len(batches)):
                acc = self.feedForward(batches[batch], predict=False)
            print(f'Epoch {epoch + 1}: {acc * 100}%')
        print('Training Finished')

    def generateOutput(self):
        'Function to use trained network to make predictions on test data'

        print('Generating Test Data Labels')

        output = self.feedForward(self.test_data)                                             #Feed Test Data to Neural Network
        pd.DataFrame.to_csv(output, 'test_predictions.csv', index=False, header=False)        #Saving Output to CSV

        print('Output File Generated')

start_time = time.time()
_, train_images_file, train_label_file, test_images_file = sys.argv
network = DigitClassifier(train_images_file, train_label_file, test_images_file)
network.train()
network.generateOutput()
print('Time taken:', round(time.time() - start_time, 2), 'seconds')