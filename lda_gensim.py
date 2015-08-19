#Latent Dirichlet Allocation for document clustering_large scale
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
import collections as co
import re
from textblob import TextBlob
from nltk.stem.porter import *
stemmer = PorterStemmer()
import numpy as np
from __future__ import division, print_function
import numpy as np
import sklearn
#import lda
#import lda.datasets
import scipy.spatial.distance as ssd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import matutils
from gensim.models.ldamodel import LdaModel #for single core processing
from gensim.models.ldamulticore import LdaMulticore #for multicore processing
import Pyro4
import multiprocessing as mp


# function : input # of document output its topic
def get_topic(n):
    doc_lda = model[doc_list[n]]    
    current_prob = 0
    for var in doc_lda:
        if var[1]>current_prob:
            current_prob = var[1]
            topic_num = var[0]
    return topic_num,re.sub('[+.0123456789\*]','',topic[topic_num])

#step 1: Import .txt files and split it into content/title
# takes several seconds
#-------------------------------
docs = []
titles = []
path = '/Users/royyang/Desktop/coffee/cof_rel_fullcontent.txt'
path1 = '/Users/royyang/Desktop/dump.txt'

path5 = '/Users/royyang/Desktop/coffee/squash_rel_fullcontent.txt'
path6 = '/Users/royyang/Desktop/coffee/apple_rel_fullcontent.txt'
path7 = '/Users/royyang/Desktop/coffee/pumpkin_rel_fullcontent.txt'
path8 = '/Users/royyang/Desktop/coffee/pears_rel_fullcontent.txt'
path9 = '/Users/royyang/Desktop/coffee/cranberries_rel_fullcontent.txt'



with open(path6) as new:
    for line in new:
        item = line.replace('\n','').split('\t')
        docs.append(re.sub('[(){}\"\[\]&#\'_0123456789;\n]', '', item[2]))
        titles.append(item[0]) 

docs[0]
res = []
for i, cont in enumerate(docs): 
    print (i)
    cont = cont.replace('\\n','').lower()
    noun = ''
    blob = TextBlob(cont)
    for var in blob.tags:
        if var[1] in ['NN'] and var[0]!='ucontent':
            noun = noun + var[0] + ' '
    res.append(noun)
    
len(res)
len(res[0])

# multiprocessing to extract noun
def extract_noun_part((m,n)):
    res = []
    for i, cont in enumerate(docs[m:n]): 
        print (i)
        cont = cont.replace('\\n','').lower()
        noun = ''
        blob = TextBlob(cont)
        for var in blob.tags:
            if var[1] in ['NN'] and var[0]!='ucontent':
                noun = noun + var[0] + ' '
        res.append(noun)
    return res

arg = spliter(len(docs),100)
start_time = time.time()
result = mp.Pool(processes=7).map(extract_noun_part, [(var[0],var[1]) for var in arg])
print("--- %s seconds ---" % (time.time() - start_time))


len(result)
len(result[0])

fin_res = []
for var in result:
    for item in var:
        fin_res.append(item)
 
 
len(item)
len(fin_res)
len(fin_res[0])       
len(result)
len(result[0][1])

res = fin_res

res[490]


# how to  train the model, after training the data is distilled only the model structure is there
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(res)
vocab = vectorizer.get_feature_names()



# jsut calculate score, use sparse matrix is very very import


# spliter fucntion, input [total number of files,how many parts to split]
def spliter(num,n_jobs):
    unit = num//n_jobs
    arg = []
    for i in range(n_jobs):
        start = i*unit
        end = (i+1)*unit
        if i==n_jobs-1:
            arg.append([start,num])
            break
        arg.append([start,end])
    return arg

arg = spliter(len(titles),1000)




# take 1h to finish if not multiprocessing
start_time = time.time()
for i in range(30):
    print (i)
    var = cosine_similarity(X[arg[i][0]:arg[i][1]],X)
    for index_a,item in enumerate(var):
        current_score = 0
        for j,each in enumerate(item):
            if each >= current_score and each < 0.999:
                current_score = each
                index_b = j
        print ([current_score,arg[i][0]+index_a,index_b])
print("--- %s seconds ---" % (time.time() - start_time))


# multiprocessing function take 600seconds to calculate 100k articles
def cos_simi_part((m,n)):
    result = []    
    var = cosine_similarity(X[m:n],X)
    for index_a,item in enumerate(var):
        current_score = 0
        for j,each in enumerate(item):
            if each >= current_score and each < 0.999:
                current_score = each
                index_b = j
        result.append([current_score,m+index_a,index_b])
    return result
    
#cos_simi_part((arg[0][0],arg[0][1]))
#    
#start_time = time.time()
#pool = mp.Pool(processes=7)
#result = pool.map(cos_simi_part, [(arg[0][0],arg[0][1]),(arg[1][0],arg[1][1]),(arg[2][0],arg[2][1])])
##pool.close()
##pool.join()
#print("--- %s seconds ---" % (time.time() - start_time))
 
start_time = time.time()
result = mp.Pool(processes=7).map(cos_simi_part, [(var[0],var[1]) for var in arg])
print("--- %s seconds ---" % (time.time() - start_time))


fin_res = []
for var in result:
    for item in var:
        print (item)
        fin_res.append(item)
        
len(fin_res)

path2 = '/Users/royyang/Desktop/ls_similarity_squash.txt'
with open(path2,'w') as new:
    for line in fin_res:
        new.write(str(line[0])+'\t'+str(line[1])+'\t'+titles[line[1]]+'\t'+str(line[2])+'\t'+titles[line[2]]+'\n')


len(titles)        


vocab
 

           
#    print ([sec_lar,i,int(np.where(var == sec_lar)[0])])

start_time = time.time()
score = cosine_similarity(X[0:10],X)
print("--- %s seconds ---" % (time.time() - start_time))


# calculate similarity and output title and error
x_array = X.toarray()
len(x_array[0])
len(X)


#Train LDA model Take 327 seconds to train the model
start_time = time.time()
model = LdaMulticore(
                    matutils.Sparse2Corpus(X,documents_columns=False), 
                    num_topics=7,passes=10,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]),
                    workers=7,
                    )
print("--- %s seconds ---" % (time.time() - start_time))



# Get all topics from training 
doc_list = []
for var in matutils.Sparse2Corpus(X,documents_columns=False):
    doc_list.append(var)

topic = model.print_topics(num_topics=7, num_words=10)

fin_sum = []
for i in range(len(doc_list)):
    fin_sum.append(get_topic(i)[0])
topic_count = co.Counter(fin_sum)

for i,var in enumerate(topic):
    [i,str(re.sub('[+.0123456789\*]','',var)),topic_count[i]]

    

# [topic,topic_words,doc_title]
for i in range(100):
    [get_topic(i),titles[i]]
 
#help(model)

#m = 1
#n = 3
#
#test = model[doc_list[m]]
#topic_s1 = []
#for var in test:
#    topic_s1.append(var[1])
#    
#
#test = model[doc_list[n]]
#topic_s2 = []
#for var in test:
#    topic_s2.append(var[1])    
#
#cosine_similarity(topic_s1,topic_s2)
#cosine_similarity(X[m],X[n])
#titles[m]
#titles[n]
#
#len(doc_list)

# method 1
start_time = time.time()
for i, var in enumerate(score):
    sec_lar = sorted(var)[-2]
    j = int(np.where(var == sec_lar)[0])
    [int(score[i][j]*100)/100,i,titles[i],get_topic(i)[0],j,titles[j],get_topic(j)[0]]
print("--- %s seconds ---" % (time.time() - start_time))    
        
len(score[0])





