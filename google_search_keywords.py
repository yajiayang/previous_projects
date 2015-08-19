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



#how to handle unicode in python
#re.sub(r'[^\x00-\x7F]+',' ', i['description']

mongo = MongoClient('cme3-mongo01.las.qa')
mongo_db = mongo['cme']
db = mongo_db['document']

#_id='www.livestrong.com/article/557524-one-great-answer-what-is-the-best-way-to-determine-how-many-calories-i-should-eat/'    
#_id = 'www.livestrong.com/article/557484-one-great-answer-how-can-i-lose-excess-skin-after-weight-loss/'
#_id = 'www.livestrong.com/article/1011048-livestrong-success-story-robin-wilson/'
#_id = 'www.livestrong.com/article/487748-healthy-her-own-terms-interview-kimberly-fowler/'
#_id = 'www.livestrong.com/blog/2-new-tech-gadgets-help-relax-sleep/'

#b=db.find({'owning_domain':'www.livestrong.com','_type':'document','type':'Article',
#          'content_type':{'$ne':None,'$nin':['Video','SlideShow','author','category']}},{'_id':1,
#'description':1,'content_type':1,'sections.steps.content':1,'sections.steps.paragraph':1,'content':1,'content':1,'tips':1,'warnings':1})


#==============================================================================
# step 1: dowanload all ehow articles and store in a txt file
#==============================================================================
b = db.find({'owning_domain':'www.ehow.com','_type':'document','type':'Article'},{'_id':1,'description':1,'content_type':1,'sections.steps.content':1,'sections.steps.paragraph':1,'warnings':1,'content':1,'tips':1,'sections.steps.paragraph':1})
# for ehow
path  = '/Users/royyang/Desktop/trending_project/dump.txt'
# for livestrong
path = '/Users/royyang/Desktop/dump/dump.txt'
count=0
with open(path,'w') as new:
       for i in b:
           print (count)
           count+=1
#           new.write(i['_id']+'\t'+(' '.join(re.sub(r'[^\x00-\x7F]+',' ', i['description']).split()) if ('description' in i) else ' ')+'\t'+
#           (' '.join(str(i['sections']).split()) if ('sections' in i) else ' ')+'\t'+
#           (' '.join(str(i['tips']).split()) if ('tips' in i) else ' ')+'\t'+
#           (' '.join(str(i['warnings']).split()) if ('warnings' in i) else ' ')+'\n')
           new.write('{}\t{}\t{}\t{}\t{}\n'.format(
                      i['_id'],
                    (' '.join(re.sub(r'[^\x00-\x7F]+',' ', i['description']).split()) if ('description' in i) else ' '),
                    (' '.join(str(i['sections']).split()) if ('sections' in i) else ' '),
                    (' '.join(str(i['tips']).split()) if ('tips' in i) else ' '),
                    (' '.join(str(i['warnings']).split()) if ('warnings' in i) else ' ')
                    ))
                


docs = []
titles = []           
with open(path) as new:
    for i,line in enumerate(new):
        print (i)
        item = line.replace('\n','').split('\t')
        text = ' '.join(item[1:]).replace('u\'','').replace('quot','')
        docs.append(re.sub('[:(){}\"\[\]&#_0123456789;\n\']', '',text))
        titles.append(item[0]) 

titles[0]
docs[0]
len(docs)
len(titles)


#==============================================================================
# # multiprocessing to extract noun, roughly 1000articles/s,835seconds to finish 700k articles
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

arg = spliter(len(docs),1000)
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
start_time = time.time()
model = LdaMulticore(
                    matutils.Sparse2Corpus(X,documents_columns=False), 
                    num_topics=9,passes=10,
                    chunksize=5000,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]),
                    workers=7,
                    )
print("--- %s seconds ---" % (time.time() - start_time))
fname = '/Users/royyang/Desktop/trending_project/re_categorization_ls/LDA_9topics'
model.save(fname)

#Load a pretrained model
model = LdaModel.load(fname, mmap='r')
type(model)

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

topic = model.print_topics(num_topics=9, num_words=50)

# store topic with probability
with open('/Users/royyang/Desktop/trending_project/re_categorization_ls/topic_with_prob.txt','w') as new:
    for i in range(9):
        new.write('{}\t{}\n'.format(str(i),topic[i]))
        

fin_sum = []
for i in range(len(doc_list)):
    fin_sum.append(get_topic(i)[0])
topic_count = co.Counter(fin_sum)

#path = '/Users/royyang/Desktop/trending_project/re_categorization_ehow/top_words_28topics.txt'
path = '/Users/royyang/Desktop/trending_project/re_categorization_ls/top_words_9topics.txt'


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
# generate all urls under each topic
#==============================================================================

tem_list = []
for i in range(len(doc_list)):
    tem = '{},{}'.format(titles[i],fin_sum[i]).split(',')
    tem[1] = float(tem[1])
    tem_list.append(tem)
tem_list = sorted(tem_list, key=lambda res: res[1],reverse=False)


with open('/Users/royyang/Desktop/trending_project/re_categorization_ls/url_topic.txt','w') as new:
    for i,var in enumerate(tem_list):
        new.write('{}\t{}\n'.format(var[0],int(var[1])))

for i in range(9):
    count = 0
    with open('/Users/royyang/Desktop/trending_project/re_categorization_ls/topic_'+str(i)+'.txt','w') as new:
        with open('/Users/royyang/Desktop/trending_project/re_categorization_ls/url_topic_sorted.txt') as old:
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
    

























