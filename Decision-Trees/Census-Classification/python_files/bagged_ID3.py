'''
Bagged Decision Trees using scikit-learn and python 2.7

Authors: Sam Castle and Alden Hart

16 March 2014
'''

import time
import random
from sklearn.ensemble import RandomForestClassifier as RFClassifier

data_file = 'census-income.data'
start = time.time()

def read_in(file_name):
   '''Reads in the data, storing it as a list of lists'''
   output_list = []
   with open(file_name) as input_file:
       for line in input_file:
           output_list.append(line.split(','))

   return output_list

def clean(lst):
   '''Removes the newline and the instance weight from the dataset'''
   for row in lst:
       row[-1] = row[-1][:-2]
       row.pop(24)

def find_mode(attribute_index, data):
   '''Finds the mode of an attribute'''
   counts = {}
   for row in data:
       if row[attribute_index] in counts:
           counts[row[attribute_index]] += 1
       else:
           if row[attribute_index] != ' ?':
               counts[row[attribute_index]] = 1

   atts = list(counts.keys())
   num = list(counts.values())

   mode = atts[num.index(max(num))]
   return mode

#find the attributes which are discrete
def find_discrete_values(dataset):
   indices = []
   i = 0
   for att in dataset[0]:
      try:
         float(att) #will produce an error for discrete attributes
      except ValueError:
         indices.append(i)
      i += 1
   return indices
         

#convert the specified indices to arbitrary numerical values
def convert(dataset, indices):
   for i in indices:
      tag = 0
      mapping = {}
      for item in dataset:
         if item[i] not in mapping:
            mapping[item[i]] = tag
            tag += 1
         item[i] = mapping[item[i]]

def impute(data):
   '''Imputes missing values with the mode of that attribute'''
   for i in range(len(data[0])):
       mode = find_mode(i, data)
       for row in data:
           if row[i] == ' ?':
               row[i] = mode

def split_labels(attributes, classifications):
   '''Separates the data into two lists, one of features and one of classifications'''
   for row in attributes:
       classifications.append(row.pop())

## Read in data

training = read_in(data_file)
print 'read in training', time.time() - start

## Clean data

clean(training)
print 'cleaned training', time.time() - start

## Fill in missing values

impute(training)
print 'imputed training', time.time() - start

## Set validation set, final test set

##validation_size = 30     ## This is the percentage of the training set to be used as validation data

test_file = 'census-income.test'
test = read_in(test_file)
print 'read in test', time.time() - start

clean(test)
print 'cleaned test', time.time() - start

impute(test)
print 'imputed test', time.time() - start

dv = find_discrete_values(training)
convert(training, dv)
convert(test, dv)


'''
validation = []
validation_len = int(len(training) * validation_size/100.0)

for i in xrange(validation_len):
   to_add = training.pop(random.randrange(len(training)))
   validation.append(to_add)

print 'made validation', time.time() - start
'''

## Split classifications from features

classifications = []
split_labels(training, classifications)
test_classes = []
split_labels(test, test_classes)

## Get random subset for bagging from remaining data
## - Probably want to vary size of random subset and generate curves to pick best
##   one

'''
for example in training:
   for att in example:
      try:
         float(att)
      except ValueError:
         print(att)
'''

print 'baseline from test = ', float(test_classes.count(0))/len(test_classes)
print 'baseline from training  = ', float(classifications.count(0))/len(classifications)

num_tests = 50
print 'n_estimators,test_accuracy,training_accuracy'
for i in range(1,num_tests+1):
   bagging = RFClassifier(n_estimators=i)
   bags = bagging.fit(training, classifications)
   test_accuracy = bags.score(test, test_classes)
   training_accuracy = bags.score(training,classifications)
   print i,',',test_accuracy,',',training_accuracy
