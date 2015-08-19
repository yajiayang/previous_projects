# get the url of a image
import nltk
import textblob
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
import pymongo
from pymongo import MongoClient# Comes with pymongo
import re
import unicodedata
import collections as co


mongo = MongoClient('cme3-mongo01.las.qa')
mongo_db = mongo['cme']
db = mongo_db['document']

#,'_archive.image.caption':{'$regex':'coffee'}
# there are 4305 images in livestrong
# there are 1524310 images in ehow redirection

ls = db.find(
                    {'owning_domain':'www.livestrong.com', '_type':"redirection", '_archive.image.url':{'$exists':1},'_archive.image.caption':{'$exists':1}},
                    {'_id':0,'_archive.image.caption':1,'_archive.image.url':1}
                    )

eh = db.find(
                    {'owning_domain':'www.ehow.com', '_type':"redirection", '_archive.image.url':{'$exists':1},'_archive.image.caption':{'$exists':1}},
                    {'_id':0,'_archive.image.caption':1,'_archive.image.url':1}
                    )
                    
#eh
                    
path = '/Users/royyang/Desktop/url_caption_ls.txt'
with open(path,'w') as new:
    for i, var in enumerate(ls):
        print (i)
        new.write(var['_archive']['image']['url']+'\t'+' '.join(re.sub(r'[^\x00-\x7F]+',' ', var['_archive']['image']['caption']).split())+'\n')


        
   




















