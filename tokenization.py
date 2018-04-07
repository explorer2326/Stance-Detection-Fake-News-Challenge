# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 07:49:38 2018

@author: Adam
"""

import nltk
sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
tokens = nltk.word_tokenize(sentence)
print(tokens)