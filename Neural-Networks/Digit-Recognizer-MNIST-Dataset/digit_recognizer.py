'''
Handwriting recognition using Neural Networks in pybrain

Authors: Sam Castle and Alden Hart

3/26/2014
'''

import csv
import time
from pybrain.datasets import SupervisedDataSet as SDS
#from pybrain.datasets import ClasificationDataset as CDS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

start = time.time()
TRAINING = 'train.csv'
TEST = 'test.csv'

def read_in_csv(filename):
    '''Reads in the .csv file containing data

    parameter: filename, the name of the .csv file
    returns: a list of lists containing the data'''
    data = []
    with open(filename, 'rb') as input_data:
        filereader = csv.reader(input_data, delimiter = ',')
        for row in filereader:
            data.append(row)

    return data

def make_pybrain_dataset(data, labels):
    '''Converts a list of lists into a pybrain dataset object

    parameter: data, as a list of lists
                labels, the labels of the datasets 
    returns: a pybrain dataset object'''

    dataset = SDS(len(data[0]), 10)
#    dataset = CDS(len(data[0]))
    for i in range(len(data)):
        num = labels[i]
        label_vec = []
        for j in range(10):
            if j == num:
                label_vec.append(1)
            else:
                label_vec.append(0)
            print num, j, labels
        print num, label_vec, labels[i], i
        dataset.addSample(data[i], label_vec)

    return dataset

def make_network(num_inputs, hidden_layers, num_outputs):
    '''Makes a pybrain neural net with the following parameters:
        - the number of input nodes
        - the number of hidden layers
        - the number of output nodes

    parameters: see above
    returns: a neural network object'''
    
    net = buildNetwork(num_inputs, hidden_layers, num_outputs)
    return net

def build_trainer(network, dataset):
    '''Trains a pybrain neural network on a given dataset

    parameters:
    network, the neural network made by pybrain
    dataset, the data to train the network on

    returns: the trainer'''
    
    trainer = BackpropTrainer(network, dataset)
    return trainer

training_raw = read_in_csv(TRAINING)
training_raw.pop(0)
print 'read in training', time.time() - start

test_raw = read_in_csv(TEST)
test_raw.pop(0)
print 'read in test', time.time() - start

labels_raw = []

for row in training_raw:
    label = row.pop(0)
    labels_raw.append(label)

##outs = {}
##for label in labels_raw:
##    if label not in outs:
##        outs[label] = 1
##    else:
##        outs[label] += 1
##
##for num in outs:
##    print num, outs[num]

print 'made labels', time.time() - start

training = make_pybrain_dataset(training_raw, labels_raw)
#training._convertToOneOfMany() # Encodes class with one output neuron per class
                                 # This makes our labels 10-dimensional 
print 'made pybrain dataset', time.time() - start

net = make_network(len(training_raw[0]), 2, 10)
print 'made network', time.time() - start

trainer = build_trainer(net, training)
print 'built trainer', time.time() - start

for i in range(10):
    error = trainer.train()
    print error
    
print 'trained network', time.time() - start

'''
Need to do:
    - Split training into cross-validation equivalence classes
    - Cross validate, but for what? What can we tune?
        - Is this just to get results
    - Get an output
        - Possibly interpret this output as a number
        - Maybe CalculateModuleOutput
    
'''
