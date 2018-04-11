# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:03:53 2018

@author: Adam
"""

#%%
import os
import sys
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import math
import calculation
from dataset import DataSet
from gensim import models

dataset = DataSet()
corpus = []
#load article body into corpus
for ID in dataset.articles:
    # raw data loading
    raw_sentences = dataset.articles[ID]
    # tokenization
    word_tokens = word_tokenize(raw_sentences)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
   
    corpus.append(filtered_sentence)

# load trained word2vec model for the purpose of vocab construction

model = gensim.models.Word2Vec.load('data/word2vec_model')
#%% 
''' 
#count total word in collection
collection_word_count = 0
for index in range(len(corpus)):
    collection_word_count = collection_word_count + len(corpus[index])

#count vocab size
print(len(model.wv.vocab)) ... 5883
'''

#%%

#count number of docs containing certain word
def get_doc_num(word):
    counter = 0
    for ID_art in range(len(corpus)):
        if corpus[ID_art].count(word) != 0:
            counter = counter + 1
    return counter

#calculate idf and store into dictionary
doc_num = 1683
idf_dict = dict.fromkeys([0])
for word, vocab_obj in model.wv.vocab.items():
    idf = math.log(doc_num/get_doc_num(word))
    idf_dict[word] = idf
    
np.save('data/idf_dict.npy', idf_dict)
#%%
#tf_idf calculation
def get_tf_idf(word,doc):
    #load existing idf dictionary extracted from training set
    #idf_dict = np.load('data/idf_dict.npy')
    doc_num = 1683
    
    #tf_idf = (doc.count(word)/len(doc))*idf_dict[word]
    tf_idf = (doc.count(word)/len(doc))*idf_dict.get(word)
    return tf_idf
    
#%%
   
# tf-idf vector for article body
tf_idf_body = dict.fromkeys([0])
for ID in dataset.articles:
    # raw data loading
    raw_sentences = dataset.articles[ID]
    # tf-idf calculation
    tf_idf_vec = []
    for word, vocab_obj in model.wv.vocab.items():
        tf_idf = get_tf_idf(word,raw_sentences)
        tf_idf_vec.append(tf_idf)
   
    tf_idf_body[ID] = tf_idf_vec
np.save('data/tf_idf_body.npy', tf_idf_body)
#%%

#tf-idf for headline
tf_idf_headline = dict.fromkeys([0])
count = 0
for s in dataset.stances:    
    count = count+1
    # raw data loading
    raw_sentences = s['Headline']
    # tf-idf calculation
    tf_idf_vec = []
    for word, vocab_obj in model.wv.vocab.items():
        tf_idf = get_tf_idf(word,raw_sentences)
        tf_idf_vec.append(tf_idf)
        
    tf_idf_headline[count] = tf_idf_vec
np.save('data/tf_idf_headline.npy', tf_idf_headline)
#%%
# calculate cosine similiarity
tf_idf_similarity = dict.fromkeys([0])
counter = 0
for s in dataset.stances:
    s['Body ID'] = int(s['Body ID'])
    counter = counter+1
    sim = calculation.get_cosine(tf_idf_headline.get(counter), tf_idf_body.get(s['Body ID']))
    tf_idf_similarity[counter] = sim
np.save('data/tf_idf_similarity.npy', tf_idf_similarity)