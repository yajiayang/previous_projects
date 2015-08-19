# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:36:18 2015

@author: royyang
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets

path = '/Users/royyang/Desktop/trending_project/coffee.txt'
res,title = [],[]
with open(path) as new:
    next(new)
    for line in new:
        item = line.replace('\n','').split(';')[-1].split(',')
        other_item = line.replace('\n','').split(';')[:-1]
        print item
        res.append([float(var) for var in item])
        title.append(other_item)
X = np.array(res)
y = np.array(title)[:,0]

np.random.seed(5)





estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=8, n_init=1,
                                              init='random')}
                                              
est = KMeans(n_clusters=10)                                              
est.fit(X)
labels = est.labels_
est.score

test_list = []
for i,var in enumerate(labels):
    print i,var
    test_list.append([y[i],var])
    
values = set(map(lambda x:x[1], test_list))
newlist = [[d[0] for d in test_list if d[1]==x] for x in values]

for var in newlist:
    print var
    

import nltk
nltk.download()
.corpus
import collections as co   
a = np.array(title)[:,0]
b = ' '.join(a).split()
co.Counter(b)

    
    
    









                            