#==============================================================================
# #Latent Dirichlet Allocation for document clustering
# Please be noted that this method can only be used for fast prototype
# the input has to be integer np.array
# if the matrix is too large(~100k), the model won't work
# the solution is gensim.ldamulticore
#==============================================================================
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
import lda
import lda.datasets
import scipy.spatial.distance as ssd
from sklearn.feature_extraction.text import TfidfVectorizer



#step 1: Import .txt files and split it into content/title
# takes several seconds
#-------------------------------
docs = []
titles = []
path = '/Users/royyang/Desktop/coffee/cof_rel_fullcontent.txt'
path1 = '/Users/royyang/Desktop/dump.txt'
with open(path) as new:
    for line in new:
        item = line.replace('\n','').split('\t')
        docs.append(re.sub('[(){}\"\[\]&#\'_0123456789;\n]', '', item[2]))
        titles.append(item[0])
#-------------------------------



#step 2: create document-term matrix   
#-------------------------------
def fn_tdm_df(docs, xColNames = None, **kwargs):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''

    #initialize the  vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    #create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames
    return df


# takes 10 min if not multiprocessing, only use noun to construct matrix    
res = []
i=0
for cont in docs: 
    print (i)
    i += 1
    cont = cont.replace('\\n','').lower()
    noun = ''
    blob = TextBlob(cont)
    for var in blob.tags:
        if var[1] in ['NN'] and var[0]!='ucontent':
            noun = noun + var[0] + ' '
    res.append(noun)
#    if i>=1000:
#        break
#--------------------------------


#step 3: prepare proper format of data for LDA model 
#--------------------------------
# prepare documment-term matrix
a = fn_tdm_df(res)
train = np.array(a.transpose())
X = train
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))


# prepare vocabulary
#type(a)
vocab = a.transpose().columns.values
#vectorizer = TfidfVectorizer()
#vectorizer.fit_transform(res)
#vocab = vectorizer.get_feature_names()
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))


# prepare titles
type(titles)
len(titles)
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))
#--------------------------------

#step 3: train LDA model 
#----------------------------------
model = lda.LDA(n_topics=5, n_iter=200, random_state=1)
model.fit(X)
#----------------------------------

#step 4: output LDA results
#----------------------------------

# output top words for each topic
topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

n = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
    
#output topic for each article
doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))    
    
for n in range(10):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}\n{}...".format(n,
                                            topic_most_pr,
                                            titles[n]))
                                            
# [topic,topic_title, # of articles]
fin_sum = []
for n in range(len(titles)):
    fin_sum.append(doc_topic[n].argmax())

topic_count = co.Counter(fin_sum)  

n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    [i,' '.join(topic_words),topic_count[i]]
    
#[topic, article_title, article_vector]
sim_vec = []
for i in range(len(titles)):
    sim_vec.append([doc_topic[i].argmax(),titles[i],train[i]])

# 1 v.s. All cosine similarity
topic_error_count = 0
for key in range(100):
    min_score = 1
    result = []
    for i in range(len(titles)):
        current_score = ssd.cosine(train[key], train[i])
        if  current_score < min_score and current_score >= 0.01:
            min_score = current_score
            result = [min_score,sim_vec[key][0],sim_vec[key][1],sim_vec[i][0],sim_vec[i][1]]
    if result[1] != result[3]:
        topic_error_count += 1
    result
topic_error_count 
        
















            
   