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
        self.hidden_units = 128
        self.outputs = 10
        self.epochs = 50
        self.batch_size = 2000
        self.learning_rate = 0.0001       #0.01 (momentum)
        # self.momentum = 0.9
        self.rms = 0.999

        #Initializing Weights
        # self.weights_hidden = pd.DataFrame(np.random.uniform(-1, 1, (self.inputs + 1, self.hidden_units + 1)) * np.sqrt(1/(self.inputs + 1 + self.hidden_units + 1)))       # +1 for bias
        # self.weights_output = pd.DataFrame(np.random.uniform(-1, 1, (self.hidden_units + 1, self.outputs)) * np.sqrt(1/(self.hidden_units + 1 + self.outputs)))
        self.weights_hidden = pd.DataFrame(np.random.uniform(-1, 1, (self.inputs + 1, self.hidden_units + 1)) / np.sqrt((self.inputs + 1) * (self.hidden_units + 1)))       # +1 for bias
        self.weights_output = pd.DataFrame(np.random.uniform(-1, 1, (self.hidden_units + 1, self.outputs)) / np.sqrt((self.hidden_units + 1) * self.outputs))
        # self.momentum_hidden = np.zeros((self.inputs + 1, self.hidden_units + 1))
        # self.momentum_output = np.zeros((self.hidden_units + 1, self.outputs))
        self.rms_hidden = np.zeros((self.inputs + 1, self.hidden_units + 1))
        self.rms_output = np.zeros((self.hidden_units + 1, self.outputs))

        print('Neural Network Initialized')

        #Importing training data and labels
        with open(train_images_file) as f:
            self.train_data = pd.read_csv(f, header=None)

        with open(train_label_file) as f:
            labels = pd.read_csv(f, header=None)
            self.train_data['Label'] = labels

        self.train_data.columns = [i for i in range(self.inputs)] + ['Label']
        self.train_data.applymap(int)
        if self.train_data.shape[0] > 10000:
            self.train_data = self.train_data.sample(frac=1)[:10000]
        print('Training Data Imported')

        #Importing testing data
        with open(test_images_file) as f:
            self.test_data = pd.read_csv(f, header=None)

        self.test_data.columns = [i for i in range(self.inputs)]
        self.test_data.applymap(int)
        print('Testing Data Imported')

    def update_weights(self, update_output, update_hidden):
        'Function to update weights'

        update_output = update_output.fillna(0)                         #make sure NA's dont occur during real testing
        update_hidden = update_hidden.fillna(0)
        # self.momentum_output = (self.momentum * self.momentum_output) + (1 - self.momentum) * update_output
        # self.momentum_hidden = (self.momentum * self.momentum_hidden) + (1 - self.momentum) * update_hidden
        self.rms_output = (self.rms * self.rms_output) + ((1 - self.rms) * update_output**2)
        self.rms_hidden = (self.rms * self.rms_hidden) + ((1 - self.rms) * update_hidden**2)
        # self.weights_output -= self.learning_rate * self.momentum_output
        # self.weights_hidden -= self.learning_rate * self.momentum_hidden
        self.weights_output -= self.learning_rate * (update_output / (np.sqrt(self.rms_output) + 1e-8))
        self.weights_hidden -= self.learning_rate * (update_hidden / (np.sqrt(self.rms_hidden) + 1e-8))

    def backPropagate(self, labels, softmax_output, output_wx, activated_first_wx, first_wx, data):
        'Function to back propagate error and adjust weights'

        # cross_entropy_loss = -labels * np.log(softmax_output)           #visualize or remove

        #Output Layer Error
        error_output = (softmax_output - labels) * derivativeSoftmax(output_wx)
        update_output = activated_first_wx.T.dot(error_output)

        #Hidden Layer Error
        error_hidden = ((self.weights_output).dot(error_output.T)).T * derivativeSigmoid(first_wx)
        update_hidden = data.T.dot(error_hidden)

        self.update_weights(update_output, update_hidden)

    def feedForward(self, batch, predict = True):
        'Function to compute the output at each layer'

        data = batch.loc[:, batch.columns != 'Label'] if not predict else batch
        data[784] = [1 for _ in range(batch.shape[0])]                               #adding bias node to input layer

        #One-hot encoding output
        if not predict:
            labels = pd.get_dummies(batch['Label'])

        #1st layer
        first_wx = data.dot(self.weights_hidden)                                     #multiply by weights
        activated_first_wx = first_wx.applymap(expit)                                #pass through activation function
        activated_first_wx[self.hidden_units] = [1 for _ in range(batch.shape[0])]   #adding bias node to hidden layer

        #Output layer
        output_wx = activated_first_wx.dot(self.weights_output)                      #multiply by weights
        softmax_output = output_wx.apply(softmax, axis=1)                            #pass through activation function

        final_output = softmax_output.idxmax(axis=1)

        if predict:
            return final_output
        else:
            accuracy = (final_output==batch['Label']).mean()
            self.backPropagate(labels, softmax_output, output_wx, activated_first_wx, first_wx, data)
            return accuracy

    def train(self):
        'Function to train the network using training data'

        shuffled_train_data = self.train_data.sample(frac=1)
        batches = np.array_split(shuffled_train_data, (self.train_data.shape[0] + 1) / self.batch_size) if self.train_data.shape[0] > self.batch_size else [shuffled_train_data]
        print('Training Data Segmented')

        avg_accuracies = []
        
        print('Training Started')
        for epoch in range(self.epochs):

            accuracies = []
            for batch in range(len(batches)):
                accuracies += [self.feedForward(batches[batch], predict=False)]
            avg_accuracy = sum(accuracies)/len(accuracies) * 100
            print(f'Epoch {epoch + 1}: {avg_accuracy}%')

            #Early Stopping
            # avg_accuracies += [avg_accuracy]
            # if len(avg_accuracies) > 10:
            #     x, y, z = avg_accuracies[-1], avg_accuracies[-2], avg_accuracies[-3]
            #     if x > 93 and y > 93 and z > 93:
            #         break

        print('Training Finished')

    def generateOutput(self):
        'Function to use trained network to make predictions on test data'

        print('Generating Test Data Labels')

        output = self.feedForward(self.test_data)                                          #Feed Test Data to Neural Network
        pd.Series.to_csv(output, 'test_predictions.csv', index=False, header=False)        #Saving Output to CSV

        print('Output File Generated')

start_time = time.time()
_, train_images_file, train_label_file, test_images_file = sys.argv
network = DigitClassifier(train_images_file, train_label_file, test_images_file)
network.train()
network.generateOutput()
print('Time taken:', round(time.time() - start_time, 2), 'seconds')