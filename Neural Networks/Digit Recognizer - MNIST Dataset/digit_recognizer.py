import csv

from sklearn.neural_networks import BernoulliRBM
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model

#constants
TRAINING = 'train.csv'
TEST = 'test.csv'

def read_in_csv(filename):
    data = []
    with open(filename, 'rb') as input_data:
        filereader = csv.reader(input_data, delimiter = ',')
        for row in filereader:
            data.append(row)

    return data


train_examples = read_in_csv(TRAINING)
train_examples.pop(0) #get rid of column headers
train_labels = []
#extract labels
for row in train_examples:
    train_labels.append(row.pop(0))

#models we will use
linear_classifier = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
