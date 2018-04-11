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
    '''
    #stemming
    ps = PorterStemmer()
    stemmed_sentences =[]
    for w in filtered_sentence:    
        stemmed_sentences.append(ps.stem(w))
    '''
    
    corpus.append(filtered_sentence)

# word2vec for the purpose of vocab construction and word count

model = gensim.models.Word2Vec(corpus, size=100, window=1, min_count=10, workers=4)
#%% 
''' 
#count total word in collection
collection_word_count = 0
for index in range(len(corpus)):
    collection_word_count = collection_word_count + len(corpus[index])
    
print(collection_word_count) ... 993904

#count vocab size
print(len(model.wv.vocab)) ... 5883
'''
collection_size = 993904
# P(w|D) with dirichlet smoothing (u = 1000)
u = 1000 
# LM for article body
lm_body = dict.fromkeys([0])
for ID in dataset.articles:
    # raw data loading
    raw_sentences = dataset.articles[ID]
    # constructing language model with dirichlet smoothing
    lm = []
    for word, vocab_obj in model.wv.vocab.items():
        p = (len(raw_sentences)/(u + len(raw_sentences)))*raw_sentences.count(word)/len(raw_sentences)+(u/(u + len(raw_sentences)))*vocab_obj.count/collection_size
        lm.append(p)
   
    lm_body[ID] = lm
np.save('data/lm_body.npy', lm_body) 
#%% 
#LM for headline
lm_headline = dict.fromkeys([0])
count = 0
for s in dataset.stances:
    
    count = count+1
    # raw data loading
    raw_sentences = s['Headline']
    # constructing language model with dirichlet smoothing
    lm = []
    for word, vocab_obj in model.wv.vocab.items():
        p = (len(raw_sentences)/(u + len(raw_sentences)))*raw_sentences.count(word)/len(raw_sentences)+(u/(u + len(raw_sentences)))*vocab_obj.count/collection_size
        lm.append(p)
        
    lm_headline[count] = lm
np.save('data/lm_headline.npy', lm_headline)
#%%
# calculate KL divergence
lm_kl_divergence = dict.fromkeys([0])
counter = 0
for s in dataset.stances:
    s['Body ID'] = int(s['Body ID'])
    counter = counter+1
    kl = calculation.get_kl(lm_headline.get(counter), lm_body.get(s['Body ID']))
    lm_kl_divergence[counter] = kl
np.save('data/lm_kl_divergence.npy', lm_kl_divergence)