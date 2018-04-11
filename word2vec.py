import os
import sys
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 
import nltk
import math
import calculation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from dataset import DataSet



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
    
'''  not used
#load article headline into corpus
for s in dataset.stances:
    # raw data loading
    raw_headlines = s['Headline']
    # tokenization
    headline_tokens = word_tokenize(raw_headlines)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_headline = [w for w in headline_tokens if not w in stop_words]
    #stemming
    ps = PorterStemmer()
    stemmed_headlines =[]
    for w in filtered_headline:    
        stemmed_headlines.append(ps.stem(w))
    
    corpus.append(stemmed_headlines)
    
'''
# train word2vec

model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=10, workers=4)
model.save('data/word2vec_model')
#%% 
#calculate average vector for article body
vector_article = dict.fromkeys([0])
for ID in dataset.articles:
    # raw data loading
    raw_sentences = dataset.articles[ID]
    # tokenization
    word_tokens = word_tokenize(raw_sentences)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    vector_sum = 0
    for index in range(len(filtered_sentence)):
        if filtered_sentence[index] in model.wv.vocab:
            vector_sum = vector_sum + model.wv[filtered_sentence[index]]
    average_vector = vector_sum / len(filtered_sentence)
    vector_article[ID] = average_vector
np.save('data/vector_article.npy', vector_article)

#%%    
#calculate average vector for headline/ cosine similarity between headline and body
vector_headline = dict.fromkeys([0])
word2vec_cosine_similarity = dict.fromkeys([0])
stance_index = dict.fromkeys([0])
count = 0
for s in dataset.stances:
    s['Body ID'] = int(s['Body ID'])
    count = count+1
    # raw data loading
    raw_sentences = s['Headline']
    # tokenization
    word_tokens = word_tokenize(raw_sentences)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    vector_sum = 0
    for index in range(len(filtered_sentence)):
        if filtered_sentence[index] in model.wv.vocab:
            vector_sum = vector_sum + model.wv[filtered_sentence[index]]
    average_vector = vector_sum / len(filtered_sentence)
    if not np.any(average_vector):
        average_vector = [0] * 99 + [0.001]
    vector_headline[count] = average_vector
    sim = calculation.get_cosine(average_vector, vector_article.get(s['Body ID']))
    word2vec_cosine_similarity[count] = sim
    stance_index[count] = s['Stance']

np.save('data/vector_headline.npy', vector_headline)
np.save('data/word2vec_cosine_similarity.npy', word2vec_cosine_similarity)
np.save('data/stance_index.npy', stance_index)