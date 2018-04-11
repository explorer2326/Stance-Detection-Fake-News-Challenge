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
from testset import TestSet
'''
testset data preparation (feature extraction)
running this script may take over 10 minutes
'''

model = gensim.models.Word2Vec.load('data/word2vec_model')

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
#calculate average vector for article body
vector_article = dict.fromkeys([0])
for ID in testset.articles:
    # raw data loading
    raw_sentences = testset.articles[ID]
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
np.save('test/vector_article.npy', vector_article)

#%%    
#calculate average vector for headline/ cosine similarity between headline and body
vector_headline = dict.fromkeys([0])
word2vec_cosine_similarity = dict.fromkeys([0])
stance_index = dict.fromkeys([0])
count = 0
for s in testset.stances:
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

np.save('test/vector_headline.npy', vector_headline)
np.save('test/word2vec_cosine_similarity.npy', word2vec_cosine_similarity)
np.save('test/stance_index.npy', stance_index)
#%%
#tf_idf calculation
idf_dict = np.load('data/idf_dict.npy')
def get_tf_idf(word,doc):
    #load existing idf dictionary extracted from training set    
    doc_num = 1683
    #tf_idf = (doc.count(word)/len(doc))*math.log(doc_num/get_doc_num(word))
    tf_idf = (doc.count(word)/len(doc))*idf_dict.item()[word]
    return tf_idf
#%% preparation of tf_idf

# tf-idf vector for article body
tf_idf_body = dict.fromkeys([0])
for ID in testset.articles:
    # raw data loading
    raw_sentences = testset.articles[ID]
    # tf-idf calculation
    tf_idf_vec = []
    for word, vocab_obj in model.wv.vocab.items():
        tf_idf = get_tf_idf(word,raw_sentences)
        tf_idf_vec.append(tf_idf)
   
    tf_idf_body[ID] = tf_idf_vec
np.save('test/tf_idf_body.npy', tf_idf_body)
#%%
#tf-idf for headline
tf_idf_headline = dict.fromkeys([0])
count = 0
for s in testset.stances:    
    count = count+1
    # raw data loading
    raw_sentences = s['Headline']
    # tf-idf calculation
    tf_idf_vec = []
    for word, vocab_obj in model.wv.vocab.items():
        tf_idf = get_tf_idf(word,raw_sentences)
        tf_idf_vec.append(tf_idf)
        
    tf_idf_headline[count] = tf_idf_vec
np.save('test/tf_idf_headline.npy', tf_idf_headline)
#%%
# calculate tf_idf cosine similiarity
tf_idf_similarity = dict.fromkeys([0])
counter = 0
for s in testset.stances:
    s['Body ID'] = int(s['Body ID'])
    counter = counter+1
    sim = calculation.get_cosine(tf_idf_headline.get(counter), tf_idf_body.get(s['Body ID']))
    tf_idf_similarity[counter] = sim
np.save('test/tf_idf_similarity.npy', tf_idf_similarity)