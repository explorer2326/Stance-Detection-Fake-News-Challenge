# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 06:51:03 2018

@author: Adam
"""

import math
import numpy as np
from numpy import asarray
from scipy.special import rel_entr

def get_cosine(vec1, vec2):
     numerator = np.dot(vec1, vec2)
     sum1 = sum([vec1[x-1]**2 for x in range(len(vec1))])
     sum2 = sum([vec2[x-1]**2 for x in range(len(vec1))])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
         return 0.0
     else:
         return float(numerator) / denominator
     
def get_kl(pk, qk):

    pk = asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=0)
    qk = asarray(qk)
    qk = 1.0*qk / np.sum(qk, axis=0)
    vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=0)
    return S
    
    