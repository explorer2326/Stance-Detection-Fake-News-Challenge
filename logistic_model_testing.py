# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 02:44:51 2018

@author: Adam
"""

import numpy as np
from numpy import *  
import matplotlib.pyplot as plt 

# load pre-processed data
stance_index = np.load('test/stance_index.npy')
word2vec_cosine_similarity = np.load( 'test/word2vec_cosine_similarity.npy' )
tf_idf_similarity = np.load('test/tf_idf_similarity.npy')

weights = {'agree': matrix([[-5.19504075],[ 3.34993646],[ 2.01851963]]), 
           'disagree': matrix([[-2.44619784],[-2.44181092],[ 3.04925852]]), 
           'discuss': matrix([[-4.65319712],[ 3.05432003],[ 3.71781905]]), 
           'unrelated': matrix([[-10.83183013],[  8.76535861],[  7.9447022 ]])}

'''
linear regression: weight

logistic regression: weight
weights1 = {'agree': matrix([[-5.65977747],[4.02992926],[2.47177695]]), 
           'disagree': matrix([[-2.23089316],[-2.80358148],[3.76998622]]), 
           'discuss': matrix([[-5.88056586],[3.74073335],[ 3.97724912]]), 
           'unrelated': matrix([[-10.50391312],[8.59422134],[7.95764097]])}

weights2 = {'agree': matrix([[-5.8753015 ],[ 4.18863965],[ 2.16801721]]), 
            'disagree': matrix([[-2.12701841],[-2.85678679],[ 3.60958088]]), 
            'discuss': matrix([[-5.71835577],[ 3.64572823],[ 3.94826021]]), 
            'unrelated': matrix([[-10.33196989],[  8.38101911],[  7.8289158 ]])}
'''
#%%
#functions

# calculate the sigmoid function  
def sigmoid(x):  
    return 1.0 / (1 + exp(-x))  

def load_data():  
    test_x = []  
    test_y = []  
        
    for index in range(25413):
        stance = stance_index.item().get(index+1)
        x1 = word2vec_cosine_similarity.item().get(index+1)
        x2 = tf_idf_similarity.item().get(index+1)
        test_x.append([1.0, x1, x2])
        if stance == 'unrelated':
            test_y.append(float(0))
        else:
            test_y.append(float(1))
    return mat(test_x), mat(test_y).transpose() 

def test_first_classifier(weights, test_x, test_y):  
    
    sample_size, feature_size = shape(test_x)  
    match_num = 0  
    for i in range(sample_size):  
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  
        if predict:
            relevant_articles_index.append(i+1) 
        else:
            unrelated_articles_index.append(i+1) 
        if predict == bool(test_y[i, 0]):  
            match_num += 1  
    accuracy = float(match_num) / sample_size  
    return accuracy

def prepare_test_data(article_id):
    test_x = []
    test_y = []
    for index in range(len(article_id)):
        stance = stance_index.item().get(article_id[index])
        x1 = word2vec_cosine_similarity.item().get(article_id[index])
        x2 = tf_idf_similarity.item().get(article_id[index])
        x_vec = [1.0,x1,x2]
        test_x.append(x_vec)
        if stance == 'unrelated':
            test_y.append(float(0))
        else:
            test_y.append(float(1))
    return mat(test_x), mat(test_y).transpose()

def test_logistic_classifier(weights, test_x, test_y):  
    
    sample_size, feature_size = shape(test_x)  
    match_num = 0  
    for i in range(sample_size):  
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  
        
        if predict == bool(test_y[i, 0]):  
            match_num += 1  
    accuracy = float(match_num) / sample_size  
    return accuracy

# predict with multi-class classifier
def one_vs_all_output(x,weights):
    prediction = [sigmoid(x * weights['agree'])[0, 0],sigmoid(x * weights['disagree'])[0, 0],sigmoid(x * weights['discuss'])[0, 0]]
    if prediction.index(np.amax(prediction)) == 0:
        output = 'agree'
    elif prediction.index(np.amax(prediction)) == 1:
        output = 'disagree'
    elif prediction.index(np.amax(prediction)) == 2:
        output = 'discuss'
    return output

def test_one_vs_all(weights, x):
     
    match_num = 0   
    #load data & compare predicted result with the true one 
    for index in range(len(x)):
        stance = stance_index.item().get(x[index])
        x1 = word2vec_cosine_similarity.item().get(x[index])
        x2 = tf_idf_similarity.item().get(x[index])
        x_vec = [1.0,x1,x2]
        if one_vs_all_output(x_vec,weights) == stance:
            match_num+=1
        
    accuracy = float(match_num) / len(x)  
    return accuracy
#%%
#execution
test_x, test_y = load_data()  
relevant_articles_index = []
unrelated_articles_index = []

test_first_classifier(weights['unrelated'], test_x, test_y)
testx1 = prepare_test_data(unrelated_articles_index)[0]
testy1 = prepare_test_data(unrelated_articles_index)[1]
accuracy1 = test_logistic_classifier(weights['unrelated'],testx1, testy1)
print('first classifier accuracy:')
print(accuracy1)
accuracy2 = test_one_vs_all(weights, relevant_articles_index)
print('second classifier accuracy:')
print(accuracy2)
overall_accuracy = (len(unrelated_articles_index)*accuracy1 + len(relevant_articles_index)*accuracy2)/(len(unrelated_articles_index)+len(relevant_articles_index))
print('overall accuracy:')
print(overall_accuracy)
weighted_score = (accuracy1*25413*25 + accuracy2*(25413-len(unrelated_articles_index))*75)/(25413*25+7064*75)
print('weighted score:')
print(weighted_score)