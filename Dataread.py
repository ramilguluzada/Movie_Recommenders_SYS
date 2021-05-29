# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:30:13 2021

@author: ramil.guluzada
"""

import pandas as pd
import os

import numpy as np
import random as rd
import scipy.sparse as sp
import torch
from time import time

# os.chdir("C:\Users\RAMIL\Desktop\Neural Graph Collaborative filtering")
mat = pd.read_csv('C:\Users\RAMIL\Desktop\Neural Graph Collaborative filtering\data\ml-100k\Movielens1M_matrix.csv')



mat.head()
mat.info()

list(mat.columns.values)
mat.index.values

train = mat.iloc[:3000,:]
train.info()
test = mat.iloc[3000:,:]
test.info()
test.head()
train.tail()

# train values

train_val = train.iloc[:,1:]
columns = len(train_val.columns.values)
rows = len(train_val.index.values)

R_train = sp.dok_matrix((rows, columns), dtype=np.float32)

for i in range(columns-1):
    for j in range(rows-1):
        R_train[j,i] = train_val.iloc[j,i]
R = R_train.tolil()

test.head()

# test values

test_val = test.iloc[:,1:]
columns = len(test_val.columns.values)
rows = len(test_val.index.values)

R_test = sp.dok_matrix((rows, columns), dtype=np.float32)

for i in range(columns-1):
    for j in range(rows-1):
        R_test[j,i] = test_val.iloc[j,i]
        


