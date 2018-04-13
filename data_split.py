import os
import sys
sys.path.append(os.getcwd())
from dataset import DataSet
import random

dataset = DataSet()
headline_index = dict.fromkeys([0])
bodyid_index = dict.fromkeys([0])
stance_index = dict.fromkeys([0])
count = 0
# store the dataset into dictionary
for s in dataset.stances:
    s['Body ID'] = int(s['Body ID'])
    count = count+1
    headline_index[count] = s['Headline']
    bodyid_index[count] = s['Body ID']
    stance_index[count] = s['Stance']
    
#shuffle 1-49972 and pick the first 4997 index as validation set
    
stance_ids = list(stance_index.keys())
stance_ids.remove(0)
r = random.Random()
r.seed(1489215)
r.shuffle(stance_ids)

validation_id = stance_ids[:4997]
train_id = stance_ids[-44975:]

def ratios(id_list):
    num_agree = 0
    num_disagree = 0
    num_discuss = 0
    num_unrelated = 0
    for index in id_list:
        if stance_index[index] == 'agree':
            num_agree += 1
        if stance_index[index] == 'disagree':
            num_disagree += 1
        if stance_index[index] == 'discuss':
            num_discuss += 1
        if stance_index[index] == 'unrelated':
            num_unrelated += 1     
    print('agree rate:')
    print(num_agree/len(id_list))
    print('disagree rate:')
    print(num_disagree/len(id_list))
    print('discuss rate:')
    print(num_discuss/len(id_list))
    print('unrelated rate:')
    print(num_unrelated/len(id_list))

#execution
print('Ratios for validation set')
ratios(validation_id)
print('Ratios for training set')
ratios (train_id)