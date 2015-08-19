#==============================================================================
# # download all ehow articles
#==============================================================================
# the command to sort a txt file by column
# cat dump1.txt | sort -n -k2 > livestrong_thin.txt
# sort by reverse order
# cat cof_rel.txt | sort -rn -k2 > ls_cof_rel_sorted.txt
#import nltk
#import textblob
#from textblob import TextBlob
#from textblob.np_extractors import ConllExtractor
#import pymongo
#import re
#import unicodedata
#import collections as co
#import time
#import multiprocessing as mp
#from sklearn.feature_extraction.text import TfidfVectorizer
#from gensim.models.ldamulticore import LdaMulticore #for multicore processing
#from gensim import matutils


from pymongo import MongoClient# Comes with pymongo
import re
import collections as co
import matplotlib.pyplot as plt

mongo = MongoClient('cme3-mongo01.las.qa')
mongo_db = mongo['cme']
db = mongo_db['document']


#==============================================================================
# step 1: dowanload all ehow articles and store in a txt file
#==============================================================================
#b = db.find({'owning_domain':'www.ehow.com','_type':'document','type':'Article'},
#            {'_id':1,'fixed_category':1,'description':1,'content_type':1,'sections.steps.content':1,'sections.steps.paragraph':1,'warnings':1,'content':1,'tips':1,'sections.steps.paragraph':1})
#b = db.find_one({'owning_domain':'www.ehow.com','_type':'document','type':'Article'},
#            {'_id':1,'fixed_category':1})
b = db.find({'owning_domain':'www.ehow.com','_type':'document','type':'Article'},
            {'_id':1,'fixed_category':1})
type(b)

all_title = []
for i,var in enumerate(b):
    single_title = [str(var['_id'])]
    for item in var['fixed_category']:
        single_title.append(str(re.sub(r'[^\x00-\x7F]+',' ',item['title'])))
    all_title.append(single_title)
    if i % 10000 == 0:
        print i

#==============================================================================
# # the first level has 28 categorey
#==============================================================================
first_level_cat = co.Counter([var[1] for var in all_title])
first_level_keys = first_level_cat.keys()
sum(first_level_cat.values())
len(first_level_keys)


#==============================================================================
# # the second level has 249 category,17 articles that don't have 3 levels
#==============================================================================
len(co.Counter([var[2] for var in all_title if not len(var) <= 3]))
sum(co.Counter([var[2] for var in all_title if not len(var) <= 3]).values())
second_level_cat = {}
second_level_summary = []
for item in first_level_keys:
    single_counter = co.Counter([var[2] for var in all_title if var[1] == item and len(var) == 4])
    second_level_cat.update({item:dict(single_counter),'total_category':len(single_counter),'total_articles':sum(single_counter.values())})
    second_level_summary.append([item,len(single_counter),sum(single_counter.values())])

second_level_cat
second_level_summary

second_level_summary_sorted = sorted(second_level_summary, key=lambda res: res[2],reverse=True)


#output the summary 
with open('/Users/royyang/Desktop/trending_project/re_categorization/second_level_summary_sorted.txt','w') as new:
    for var in second_level_summary_sorted:
        new.write('{}\t{}\n'.format(var[0],var[2]))
#        new.write('{}\t{}\t{}\n'.format(var[0],var[1],var[2]))
        
#==============================================================================
# # the third level has 2857 category
#==============================================================================
third_level_cat = dict(co.Counter([var[3] for var in all_title if len(var) == 4]))
len(third_level_cat)
summary_third_level = co.Counter(third_level_cat.values())
summary_third_level_values = summary_third_level.values()
summary_third_level_keys = summary_third_level.keys()
plt.plot(summary_third_level_keys[:50],summary_third_level_values[:50])







           