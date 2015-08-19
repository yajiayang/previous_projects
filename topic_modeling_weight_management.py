#==============================================================================
# # download all ehow articles
#==============================================================================
# the command to sort a txt file by column
# cat dump1.txt | sort -n -k2 > livestrong_thin.txt
# sort by reverse order
# cat cof_rel.txt | sort -rn -k2 > ls_cof_rel_sorted.txt
import nltk
import textblob
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
import pymongo
from pymongo import MongoClient# Comes with pymongo
import re
import unicodedata
import collections as co
import time
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.ldamulticore import LdaMulticore #for multicore processing
from gensim import matutils
from gensim.models import LdaModel
import matplotlib.pyplot as plt


#==============================================================================
# please specify topic number
#==============================================================================
topic_number = 5


#==============================================================================
# #
#generate subdumps based on topic models
#==============================================================================
path = '/Users/royyang/Desktop/trending_project/re_categorization_ls/ls.txt'

docs = []
with open(path) as origin:
    for line in origin:
        docs.append(line.strip('\n').split('\t'))
docs[0]
len(docs)

i = 0
with open('/Users/royyang/Desktop/trending_project/re_categorization_ls/topic_1_full.txt','w') as output:
    with open('/Users/royyang/Desktop/trending_project/re_categorization_ls/topic_1.txt') as new:
        for line in new:
            url = line.strip('\n')
            for var in docs:
                if url == var[0]:
                    print i
                    i += 1
                    output.write('{}\t{}\n'.format(var[0],var[1]))
                        

#==============================================================================
# starting from a dumpfile
#==============================================================================
folder_name = '/Users/royyang/Desktop/trending_project/weight_management/'
file_name = 'weight_full.txt' 
docs = []
titles = []           
with open(folder_name+file_name) as new:
    for i,line in enumerate(new):
        print (i)
        item = line.replace('\n','').split('\t')
        text = ' '.join(item[1:]).replace('u\'','').replace('quot','')
        docs.append(re.sub('[:(){}\"\[\]&#_0123456789;\n\']', '',text))
        titles.append(item[0]) 

titles[0]
docs[2]
len(docs)
len(titles)


#==============================================================================
# # multiprocessing to extract noun, roughly 1000articles/s,5seconds to finish 15k articles
#==============================================================================
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

def extract_noun_part((m,n)):
    res = []
    for i, cont in enumerate(docs[m:n]): 
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
 
res = fin_res
len(res)
res[0]
#==============================================================================
# #Train LDA model Take 1655 seconds to train the model
#==============================================================================
# No need to run LDA everytime, model has bee stored
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(res)
vocab = vectorizer.get_feature_names()


# single LDA
topic_number = 15
start_time = time.time()
model = LdaMulticore(
                    matutils.Sparse2Corpus(X,documents_columns=False), 
                    num_topics=topic_number,passes=10,
                    chunksize=5000,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]),
                    workers=7,
                    )
print("--- %s seconds ---" % (time.time() - start_time))
fname = folder_name+'LDA'+str(topic_number)+'topics'
model.save(fname)

#Load a pretrained model
model = LdaModel.load(fname, mmap='r')
type(model)

#perplexity
perplexity = model.log_perplexity(matutils.Sparse2Corpus(X,documents_columns=False), total_docs=None)



# batch LDA
model_eval = []
for k in range(2,21):
    topic_number = k
    start_time = time.time()
    model = LdaMulticore(
                        matutils.Sparse2Corpus(X,documents_columns=False), 
                        num_topics=topic_number,passes=10,
                        chunksize=5000,
                        id2word=dict([(i, s) for i, s in enumerate(vocab)]),
                        workers=7,
                        )
    print("--- %s seconds ---" % (time.time() - start_time))
#    fname = folder_name+'LDA_9topics'
#    model.save(fname)
#    
#    #Load a pretrained model
#    model = LdaModel.load(fname, mmap='r')
#    type(model)
    
    #perplexity
    perplexity = model.log_perplexity(matutils.Sparse2Corpus(X,documents_columns=False), total_docs=None)
    model_eval.append([topic_number,perplexity])

plt.plot([var[0] for var in model_eval],[var[1] for var in model_eval])
#==============================================================================
# store the log perplexity from 2 to 20 topics
#==============================================================================
with open(folder_name+'perplexity_score_under_20.txt','w') as new:
    for var in model_eval:
        new.write('{}\n'.format(var))



#==============================================================================
# # Get all topics from training 
# topic_number, number_of_aritcles, top_words
#==============================================================================
def get_topic(n):
    doc_lda = model[doc_list[n]]    
    current_prob = 0
    for var in doc_lda:
        if var[1]>current_prob:
            current_prob = var[1]
            topic_num = var[0]
    return topic_num,re.sub('[+.0123456789\*]','',topic[topic_num])

doc_list = []
for var in matutils.Sparse2Corpus(X,documents_columns=False):
    doc_list.append(var)

topic = model.print_topics(num_topics=topic_number, num_words=50)

# store topic with probability
with open(folder_name+'topic_with_prob_'+str(topic_number)+'_topics.txt','w') as new:
    for i in range(topic_number):
        new.write('{}\t{}\n'.format(str(i),topic[i]))
        

fin_sum = []
for i in range(len(doc_list)):
    fin_sum.append(get_topic(i)[0])
topic_count = co.Counter(fin_sum)

#path = '/Users/royyang/Desktop/trending_project/re_categorization_ehow/top_words_28topics.txt'
path = folder_name+'top_words_for_'+str(topic_number)+'_topics.txt'


tem_list = []
for i,var in enumerate(topic):
    tem = '{},{},{}'.format(str(i),topic_count[i],str(re.sub('[+.0123456789\*]','',var))).split(',')
    tem[1] = float(tem[1])
    tem_list.append(tem)
tem_list = sorted(tem_list, key=lambda res: res[1],reverse=True)
    
with open(path,'w') as new:
    for i,var in enumerate(tem_list):
        new.write('{}\t{}\t{}\n'.format(var[0],int(var[1]),var[2]))

                    
#==============================================================================
# LDA visulization
#==============================================================================
#heatmap
from bokeh.charts import HeatMap, output_file, show
from bokeh.palettes import YlOrRd9 as palette
from bokeh.sampledata.unemployment1948 import data
import collections as co
import pandas as pd



# store topic with probability

# construct corpus
index_all = []
with open(folder_name+'topic_with_prob_'+str(topic_number)+'_topics.txt') as new:
    for line in new:
        item = line.strip('\n').split('\t')
        index_one = [var.split('*')[1].strip( ) for var in item[1].split('+')[:5]]
        index_all += index_one
corpus = dict(co.Counter(index_all))
corpus = dict.fromkeys(corpus, 0)

# construct value matrix
value_mt = []
with open(folder_name+'topic_with_prob_'+str(topic_number)+'_topics.txt') as new:
    for line in new:
        corpus = dict(co.Counter(index_all))
        corpus = dict.fromkeys(corpus, 0)
        item = line.strip('\n').split('\t')
        index_one = [var.split('*')[1].strip( ) for var in item[1].split('+')[:5]]
        value_one = [var.split('*')[0].strip( ) for var in item[1].split('+')[:5]]
        corpus_one = corpus
        for i,var in enumerate(index_one):
            corpus_one[var] = float(value_one[i])
        value_mt.append([int(1000*var) for var in corpus_one.values()])

df = pd.DataFrame(value_mt)
topic_words = pd.DataFrame(corpus.keys())
df2 = df.transpose().set_index(topic_words[topic_words.columns[0]])
df3 = df2.transpose()
topic_label = pd.DataFrame(['topic_'+str(i) for i in range(topic_number)])
df3 = df3.set_index(topic_label[topic_label.columns[0]])

html_path = folder_name+str(topic_number)+'_topic_Weight_Management.html'
output_file(html_path, title=str(topic_number)+'_topic_Weight_Management.html')

palette = palette[::-1]  # Reverse the color order so dark red is highest unemployment
hm = HeatMap(df3, title=str(topic_number)+'_topic_Weight_Management.html', tools = "reset,resize",
             width=1300,height=700, palette=palette)

show(hm)


#==============================================================================
# generate all urls under each topic
#==============================================================================

tem_list = []
for i in range(len(doc_list)):
    tem = '{},{}'.format(titles[i],fin_sum[i]).split(',')
    tem[1] = float(tem[1])
    tem_list.append(tem)
tem_list = sorted(tem_list, key=lambda res: res[1],reverse=False)


with open(folder_name+'url_topic.txt','w') as new:
    for i,var in enumerate(tem_list):
        new.write('{}\t{}\n'.format(var[0],int(var[1])))

for i in range(topic_number):
    count = 0
    with open(folder_name+'topic_'+str(i)+'.txt','w') as new:
        with open(folder_name+'url_topic.txt') as old:
            for line in old:
                if line.strip('\n').split('\t')[1] == str(i):
                    count += 1
                    new.write('{}\n'.format(line.strip('\n').split('\t')[0]))
    print i,count
        

#==============================================================================
# get the total list of key words from ehow.com and subtract what has been downloaded
#==============================================================================
import collections as co
result = []
total_list = ''
with open('/Users/royyang/Desktop/trending_project/top_words.txt') as new:
    for i,line in enumerate(new):
#        print (i)
        print line.replace('\n','').split('\t')[1]
        total_list += line.replace('\n','').split('\t')[1]
total_list = co.Counter(total_list.split()).keys()
len(total_list)
'agreement' in total_list 

import os
already = []
folder_path = '/Users/royyang/Desktop/Top_search_term/'
for path, subdirs, files in os.walk(folder_path):
    for i,var in enumerate(files):
        already.append(var.split('_')[-1][:-4])
already = co.Counter(already).keys()
len(already)


to_do_list = [var for var in total_list if not var in already]
len(to_do_list)


#==============================================================================
# restart the download
#==============================================================================
import top_search_scraper as tss
import time
import codecs


def scraper_one(word_list):
    r = tss.Belieber()
    r.ask(word_list)

word_list1 = 'agreement'
scraper_one(word_list1)

mp.Pool(processes=10).map(scraper_one,[var for var in to_do_list])

#==============================================================================
#how to write in unicode
#==============================================================================
a = 'agreement en español'
with codecs.open('/Users/royyang/Desktop/trending_project/test.txt','w','utf-8') as new:
    new.write(u'{}abc{}'.format(a.decode('utf-8'),a.decode('utf-8')))
    
    

    

























