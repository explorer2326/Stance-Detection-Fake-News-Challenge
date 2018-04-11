# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:04:33 2018

@author: Adam
"""
import numpy as np
from numpy import *  
import matplotlib.pyplot as plt  
import time  

# load pre-processed data
stance_index = np.load('data/stance_index.npy')
word2vec_cosine_similarity = np.load( 'data/word2vec_cosine_similarity.npy' )
tf_idf_similarity = np.load('data/tf_idf_similarity.npy')
lm_kl_divergence = np.load('data/lm_kl_divergence.npy')

#storing weights for different classifier
weights = {
		"unrelated" : [],
		"agree" : [],
		"disagree" : [],
		"discuss" : []
	} 

def load_data():  
    train_x = []  
    train_y = []  
        
    for index in range(49972):
        stance = stance_index.item().get(index+1)
        x1 = word2vec_cosine_similarity.item().get(index+1)
        x2 = tf_idf_similarity.item().get(index+1)
        train_x.append([1.0, x1, x2])
        if stance == 'unrelated':
            train_y.append(float(0))
        else:
            train_y.append(float(1))
    return mat(train_x), mat(train_y).transpose() 
#%%
# train a linear regression model using stochastic gradient descent 
def train_linear_classifier(train_x, train_y,class_name):  
    # calculate training time  
    startTime = time.time()    
    sample_size, feature_size = shape(train_x)
    # hyperparameters setting
    alpha = 0.01
    max_iteration = 100  
    weights = {
		"unrelated" : [],
		"agree" : [],
		"disagree" : [],
		"discuss" : []
	} 
    weights[class_name] = ones((feature_size, 1))    
    # optimize through gradient descent algorilthm  
    for k in range(max_iteration):  
        for i in range(sample_size):  
            output = train_x[i, :] * weights[class_name] 
            error = train_y[i, 0] - output  
            weights[class_name] = weights[class_name] + alpha * train_x[i, :].transpose() * error 
    
    print ('Training took %fs!' % (time.time() - startTime))  
    return weights[class_name] 
#%%
def test_first_classifier(weights, test_x, test_y):  
    
    sample_size, feature_size = shape(test_x)  
    match_num = 0  
    for i in range(sample_size):  
        predict = (test_x[i, :] * weights)[0, 0] > 0.5  
        
        if predict:
            relevant_articles_index.append(i+1) 
        else:
            unrelated_articles_index.append(i+1) 
        
        if predict == bool(test_y[i, 0]):  
            match_num += 1  
    accuracy = float(match_num) / sample_size  
    return accuracy

def test_linear_classifier(weights, test_x, test_y):  
    
    sample_size, feature_size = shape(test_x)  
    match_num = 0  
    for i in range(sample_size):  
        predict = (test_x[i, :] * weights)[0, 0] > 0.5  
        
        if predict == bool(test_y[i, 0]):  
            match_num += 1  
    accuracy = float(match_num) / sample_size  
    return accuracy

def load_relevant(x,stance_name):  
    train_x = []  
    train_y = []  
        
    for index in range(len(x)):
        stance = stance_index.item().get(x[index])
        x1 = word2vec_cosine_similarity.item().get(x[index])
        x2 = tf_idf_similarity.item().get(x[index])
        train_x.append([1.0, x1, x2])
        if stance == stance_name:
            train_y.append(float(1))
        else:
            train_y.append(float(0))
    return mat(train_x), mat(train_y).transpose()

# train multi-class logistic classifier (after relevant-or-not classifier)
def train_one_vs_all():
    #agree or not
    train_x, train_y = load_relevant(relevant_articles_index,'agree')  
    test_x = train_x; test_y = train_y   
    weights['agree'] = train_linear_classifier(train_x, train_y,'agree')
    accuracy = test_linear_classifier(weights['agree'], test_x, test_y)    
    print ('The accuracy for the agree-or-not classifier is: %.3f%%' % (accuracy * 100)) 
    
    #disagree or not
    train_x, train_y = load_relevant(relevant_articles_index,'disagree')  
    test_x = train_x; test_y = train_y   
    weights['disagree'] = train_linear_classifier(train_x, train_y,'disagree')
    accuracy = test_linear_classifier(weights['disagree'], test_x, test_y)    
    print ('The accuracy for the disagree-or-not classifier is: %.3f%%' % (accuracy * 100))  
    
    #discuss or not
    train_x, train_y = load_relevant(relevant_articles_index,'discuss')  
    test_x = train_x; test_y = train_y   
    weights['discuss'] = train_linear_classifier(train_x, train_y,'discuss') 
    accuracy = test_linear_classifier(weights['discuss'], test_x, test_y)    
    print ('The accuracy for the discuss-or-not classifier is: %.3f%%' % (accuracy * 100))

# predict with multi-class classifier
def one_vs_all_output(x,weights):
    prediction = [(x * weights['agree'])[0, 0],(x * weights['disagree'])[0, 0],(x * weights['discuss'])[0, 0]]
    if prediction.index(np.amax(prediction)) == 0:
        output = 'agree'
    elif prediction.index(np.amax(prediction)) == 1:
        output = 'disagree'
    elif prediction.index(np.amax(prediction)) == 2:
        output = 'discuss'
    return output

# test one-vs-all classifier
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

#%%
#first classifier
#step 1: load data  
print ("step 1: load data...")  
train_x, train_y = load_data()  
test_x = train_x; test_y = train_y  
# step 2: training first classifier (relevant or not)...  
print ("step 2: training first classifier...")   
weights['unrelated'] = train_linear_classifier(train_x, train_y,'unrelated')  
relevant_articles_index = []
unrelated_articles_index = []
# step 3: first classifier test  
print ("step 3: testing first classifier...")  
accuracy = test_first_classifier(weights['unrelated'], test_x, test_y)  
  
# step 4: first classifier result display  
print ("step 4: show the result...")    
print ('The accuracy for the unrelated-or-not classifier is: %.3f%%' % (accuracy * 100))  

#%%
# second classifier
#step 5: training 'agree', 'disagree', 'discuss' classifier... 
print ("step 6: training second classifier...")  
train_one_vs_all()

#%%
#accuracy testing
test_first_classifier(weights['unrelated'], test_x, test_y)
testx1 = prepare_test_data(unrelated_articles_index)[0]
testy1 = prepare_test_data(unrelated_articles_index)[1]
accuracy1 = test_linear_classifier(weights['unrelated'],testx1, testy1)
print('first classifier accuracy:')
print(accuracy1)
accuracy2 = test_one_vs_all(weights, relevant_articles_index)
print('second classifier accuracy:')
print(accuracy2)
overall_accuracy = (len(unrelated_articles_index)*accuracy1 + len(relevant_articles_index)*accuracy2)/(len(unrelated_articles_index)+len(relevant_articles_index))
print('overall accuracy:')
print(overall_accuracy) 