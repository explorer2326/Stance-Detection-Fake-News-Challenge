# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 08:42:58 2018

@author: Adam
"""
import numpy as np
import matplotlib

stance_index = np.load('data/stance_index.npy')
word2vec_cosine_similarity = np.load( 'data/word2vec_cosine_similarity.npy' )
tf_idf_similarity = np.load('data/tf_idf_similarity.npy')
lm_kl_divergence = np.load('data/lm_kl_divergence.npy')

#%%   
def stance_plot(id):
    stance = stance_index.item().get(id)
    if stance == 'agree':
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), tf_idf_similarity.item().get(id),c = 'r',marker='+',s=5)
    
    elif stance == 'disagree':
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), tf_idf_similarity.item().get(id),c = 'b',marker='+',s=5)
    
    elif stance == 'discuss':
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), tf_idf_similarity.item().get(id),c = 'g',marker='+',s=5)    
    '''
    else:
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), tf_idf_similarity.item().get(id),c = 'y',marker='+',s=5)
    '''

#for index in range(len(stance_index.item())):
    
for index in range(10000):
    stance_plot(index)

plt.savefig('test.jpeg',format='jpeg', dpi=1200)
#%%
def stance_plot2(id):
    stance = stance_index.item().get(id)
    if stance == 'unrelated':
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), lm_kl_divergence.item().get(id),c = 'y',s=5,marker='+')
    '''
    elif stance == 'disagree':
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), lm_kl_divergence.item().get(id),c = 'b',marker='+')
    elif stance == 'discuss':
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), lm_kl_divergence.item().get(id),c = 'g',marker='+')    
    else:
        matplotlib.pyplot.scatter(word2vec_cosine_similarity.item().get(id), lm_kl_divergence.item().get(id),c = 'y',marker='+')
    '''

#for index in range(len(stance_index.item())):
    
for index in range(49972):
    stance_plot2(index)

plt.savefig('test2.jpeg',format='jpeg', dpi=1200)