'''
Script to implement k-Nearest Neighbor algorithm.  Classifies mushrooms as edible
or inedible based on several characteristics.

Authors: Sam Castle and Alden Hart
'''

import time
start = time.time()

## Notes on the data

'''
Notes: The individual test cases are strings of the form:

0. edible? e = yes, p = poisonous
1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
4. bruises?: bruises=t,no=f 
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
6. gill-attachment: attached=a,descending=d,free=f,notched=n 
7. gill-spacing: close=c,crowded=w,distant=d 
8. gill-size: broad=b,narrow=n 
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
10. stalk-shape: enlarging=e,tapering=t 
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
16. veil-type: partial=p,universal=u 
17. veil-color: brown=n,orange=o,white=w,yellow=y 
18. ring-number: none=n,one=o,two=t 
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
'''

import math

NUMBER_OF_ATTRIBUTES = 23
MISSING_PENALTY = 3.5
DIFFERENT_PENALTY = 7

def find_mode(featureNum, classification):
   '''
   finds the mode of the supplied feature number in the training data

   Parameters: featureNum: the index of the particular feature in the mushroom data

   Returns the mode value of the feature in the training data
   '''
   featureValues = {}
   for data in training:
      if data[0] == classification and data[featureNum] != '?':
         try:
            featureValues[data[featureNum]] += 1
         except KeyError:
            featureValues[data[featureNum]] = 1

   maxValue = ""
   maxCount = 0
   for key in featureValues:
      if featureValues[key] > maxCount:
         maxValue = key
         maxCount = featureValues[key]

   return maxValue

def construct_mode_list(classification):
   '''
   Returns a list of the mode of each feature in the training data
   '''
   modeList = []
   for i in range(1, NUMBER_OF_ATTRIBUTES): #no need to find mode for attribute 0, the class
      modeList.append(find_mode(i, classification))
   return modeList

def replace_missing_values():
   '''
   Replaces all the missing values in the training data with the corresponding
   value stored in the mode list.
   '''
   modeListPoisonous = construct_mode_list('p')
   modeListEdible = construct_mode_list('e')
   for shroom in training:
      for i in range(len(shroom)):
         if shroom[i] == '?':
            if shroom[0] == 'p':
               shroom[i] = modeListPoisonous[i]
            else:
               shroom[i] = modeListEdible[i]
   

def one_d_distance(x,y):
   '''
   Finds the Euclidean distance between two items in one dimension.  If the inputs
   are identical, the distance between them is zero.  If they are not, the distance
   is 1.

   Parameters: x, y: two strings

   Retiurns: 0 if x and y are identical strings, 1 if they are not
   '''
   if x == y:
       return 0
   elif x == '?' or y == '?':
       return MISSING_PENALTY
   return DIFFERENT_PENALTY

def total_distance(x_list,y_list):
   '''
   Finds the total Euclidean distance between two lists of attributes.

   Parameters: x_list, y_list: two lists of strings

   Returns: the total distance between the two lists
   '''

   total = 0

   for i in range(1, NUMBER_OF_ATTRIBUTES):
       total += one_d_distance(x_list[i], y_list[i])

   return math.sqrt(total)

def distance_score(training, test):
   '''
   Weighs the distance between x_list and y_list according to the Gaussian equation

   Parameters: training, test: two lists of strings.
   training: the training data set
   test: the unknown data set

   Returns: the weighted distance between the two lists
   '''
   weight = math.e**(-total_distance(training, test))

   if training[0] == 'p':
       weight *= -1

   return weight

def classify_shroom(mushroom):
   '''
   Takes one mushroom from the reserve set and classifies it based on the training set

   Parameters: mushroom: a list of attributes of a mushroom

   Returns: 'e' or 'p' based on the distance score
   '''

   total = 0

   for item in training:
      if not(item is mushroom): #don't consider the same objects when cross-validating training data
         score = distance_score(item, mushroom)
         total += score

   if total < 0:
       return 'p'
   elif total > 0:
       return 'e'
   else:
       return 'Score was 0.  Crap.'



## Reads in the data

test_cases_raw = []

with open('agaricus-lepiota.data', 'r') as data:
   for line in data:
       test_cases_raw.append(line)


## Cleans data to make each test case a list of attributes

test_cases = []

for item in test_cases_raw:
   new = item.split(',')
   remove_newline = new[-1][-2]
   new = new[:-1]
   new.append(remove_newline)
   test_cases.append(new)

## Sets aside 1500 studies to test the algorithm

reserves = []
training = []
reserve_count = 0

for item in test_cases:
   if reserve_count < 1500:
       reserves.append(item)
   else:
       training.append(item)
   reserve_count += 1


def tester():
   output = {}
   output['Correct'] = 0
   output['Incorrect'] = 0

   #change this to test against training or reserves
   #remember to fill in mode of reserves
   for data in reserves: 
       result = classify_shroom(data)
       if result == data[0]:
           output['Correct'] += 1
       else:
           output['Incorrect'] += 1

   for case in output:
       print case, output[case]

replace_missing_values() #in the training data by the mode from training data

tester()

print 'Tests complete.'
print 'Runtime =', time.time() - start
