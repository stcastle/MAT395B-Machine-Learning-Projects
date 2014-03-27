
'''
Boosted Decision Trees using scikit-learn and python 2.7

Authors: Sam Castle and Alden Hart

16 March 2014
'''

import time
import random
from sklearn.ensemble import AdaBoostClassifier as ABClassifier
import csv

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

def write_to_file(filename, data):
    with open(filename, 'wb') as output:
        output_writer = csv.writer(output, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            output_writer.writerow(row)

## Read in data

training = read_in(data_file)
print 'read in training', time.time() - start

## Clean data

clean(training)
print 'cleaned training', time.time() - start

## Fill in missing values

impute(training)
print 'imputed training', time.time() - start

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


## Set validation set, final test set

validation_size = 30     ## This is the percentage of the training set to be used as validation data
validation = []
validation_len = int(len(training) * validation_size/100.0)

for i in xrange(validation_len):
  to_add = training.pop(random.randrange(len(training)))
  validation.append(to_add)

print 'made validation', time.time() - start

validation_classes = []
split_labels(validation, validation_classes)

## Split classifications from features

classifications = []
split_labels(training, classifications)
test_classes = []
split_labels(test, test_classes)


'''
#Compute baseline scores.
print 'baseline from test = ', float(test_classes.count(0))/len(test_classes)
print 'baseline from training  = ', float(classifications.count(0))/len(classifications)
print 'baseline from validation  = ', float(validation_classes.count(0))/len(validation_classes)
'''

max_iterations = 20000
boosting = ABClassifier(n_estimators=max_iterations)
print 'made clf',time.time()-start
clf = boosting.fit(training,classifications)
print 'fit clf',time.time()-start
validation_score_generator = clf.staged_score(validation, validation_classes)
training_score_generator = clf.staged_score(training, classifications)
print 'made generators',time.time()-start
print 'Reporting boosting accuracy...'
print 'num estimators , accuracy on validation, accuracy on training'

output_data = [['num estimators', 'accuracy on validation', 'accuracy on training']]
for i in range(1, max_iterations + 1):
    output_data.append([i, next(validation_score_generator), next(training_score_generator)])


#print output_data
write_to_file('adaboost_data.txt', output_data)
print 'write done'
