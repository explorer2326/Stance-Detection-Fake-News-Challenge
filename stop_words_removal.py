# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 07:35:05 2018

@author: Adam
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]


print(word_tokens)
print(filtered_sentence)