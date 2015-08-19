#ge the reference url'myfitnesspal'
import nltk
import textblob
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
import pymongo
from pymongo import MongoClient# Comes with pymongo
import re
import unicodedata
import collections as co
#how to handle unicode in python
#re.sub(r'[^\x00-\x7F]+',' ', i['description']

mongo = MongoClient('cmereporting.prod.dm.local')
mongo_db = mongo['cme']
db = mongo_db['document']

c=db.find({'owning_domain':'www.livestrong.com','_type':'document','type':'Article','references.url':{'$regex':'myfitnesspal'}},{'_id':1,'references':1})
path = '/Users/royyang/Desktop/ref_url_myfit.txt'
with open(path,'w') as new:
    for i, var in enumerate(c):
        for item in var['references']:
            if 'url' in item.keys():
                if 'myfitnesspal' in item['url']:
                    new.write(str(var['_id'])+'\t'+str(item['url'])+'\t'+str(item['title'])+'\n')


