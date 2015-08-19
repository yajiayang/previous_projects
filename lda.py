#lda
from __future__ import division, print_function
import numpy as np
import sklearn
import lda
import lda.datasets
import scipy.spatial.distance as ssd


#/Library/Python/2.7/lib/python/site-packages
# document-term matrix
#X = lda.datasets.load_reuters()
X = train
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))

# the vocab
#vocab = lda.datasets.load_reuters_vocab()
vocab
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))

# titles for each story
#titles = lda.datasets.load_reuters_titles()
titles
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))

doc_id = 0
word_id = 3117

print("doc id: {} word id: {}".format(doc_id, word_id))
print("-- count: {}".format(X[doc_id, word_id]))
print("-- word : {}".format(vocab[word_id]))
print("-- doc  : {}".format(titles[doc_id]))

model = lda.LDA(n_topics=50, n_iter=200, random_state=1)
model.fit(X)

topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

for n in range(5):
    sum_pr = sum(topic_word[n,:])
    print("topic: {} sum: {}".format(n, sum_pr))
    
n = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

    
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
    
    

# 1 v.s. All
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




# 1 v.s. All truncated version
topic_error_count = 0
path = '/Users/royyang/Desktop/coffee_article_pairs.txt'
path1 = '/Users/royyang/Desktop/coffee_article_pairs_new.txt'


with open(path,'w') as new:
    for key in range(len(titles)):
        key
        min_score = 1
        result = []
        for i in range(len(titles)):
            current_score = ssd.cosine(train[key], train[i])
            if current_score < min_score and current_score >= 0.001:
                min_score = current_score
                result = [min_score,sim_vec[key][1],sim_vec[i][1]]
        new.write(str(round(result[0],3))+'\t'+result[1]+'\t'+result[2]+'\n')
#        if result[1] != result[3]:
#            topic_error_count += 1
topic_error_count                                 
                                         
                 


                           
import matplotlib.pyplot as plt

# use matplotlib style sheet
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass

f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0, 5, 9, 14, 19]):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-50,4350)
    ax[i].set_ylim(0, 0.08)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()

f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate(range(15,20)):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 21)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()


# without 
with open(path1,'w') as new:
    with open(path) as old:
        for line in old:
            item = line.replace('\n','').split('\t')
            new.write(item[1]+'\t'+item[2]+'\n')
    
    

