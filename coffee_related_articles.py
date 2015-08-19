#==============================================================================
#find all coffee-related articles in livestrong
#in each article find the related noun phrases
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

b=db.find({'owning_domain':'www.livestrong.com','_type':'document','type':'Article',
          'content_type':{'$ne':None,'$nin':['Video','SlideShow','author','category']}},{'_id':1,
'description':1,'content_type':1,'sections.steps.content':1,'sections.steps.paragraph':1,'content':1,'content':1,'tips':1,'warnings':1})




path = '/Users/royyang/Desktop/coffee/dump.txt'
path1 = '/Users/royyang/Desktop/coffee/cof_rel.txt'
path2 = '/Users/royyang/Desktop/coffee/newac'
path3 = '/Users/royyang/Desktop/coffee/cof_rel_fullcontent.txt'
path4 = '/Users/royyang/Desktop/coffee/cof_rel_wordrank.txt'


count=0
with open(path,'w') as new:
       for i in b:
           print (count)
           count+=1
           new.write(i['_id']+'\t'+(' '.join(re.sub(r'[^\x00-\x7F]+',' ', i['description']).split()) if ('description' in i) else ' ')+'\t'+
           (' '.join(str(i['sections']).split()) if ('sections' in i) else ' ')+'\t'+
           (' '.join(str(i['tips']).split()) if ('tips' in i) else ' ')+'\t'+
           (' '.join(str(i['warnings']).split()) if ('warnings' in i) else ' ')+'\n')


# count the occurance of coffee in each article
def coffee_rel(path):
    with open(path) as handle:
        for line in handle:
            item = line.replace('\n','').split('\t')
            a = item[1]+item[2]+item[3]+item[4]
            match = re.findall('bacon',a)
            if match != []:
                yield item[0],len(match)

# output url, counts of coffee, full content
def coffee_rel_full(path):
    with open(path) as handle:
        for line in handle:
            item = line.replace('\n','').split('\t')
            a = item[1]+item[2]+item[3]+item[4]
            match = re.findall('coffee',a)
            if match != []:
                yield item[0],len(match),a
            
path5 = '/Users/royyang/Desktop/coffee/squash_rel_fullcontent.txt'
path6 = '/Users/royyang/Desktop/coffee/apple_rel_fullcontent.txt'
path7 = '/Users/royyang/Desktop/coffee/pumpkin_rel_fullcontent.txt'
path8 = '/Users/royyang/Desktop/coffee/pears_rel_fullcontent.txt'
path9 = '/Users/royyang/Desktop/coffee/cranberries_rel_fullcontent.txt'

a=coffee_rel_full(path)
count=0
with open(path3,'w') as new:
    for item in a:
        count+=1
        print (count)
        new.write(item[0]+'\t'+str(item[1])+'\t'+str(item[2])+'\n')

#extract top nouns related to coffee
count=0
noun = []
with open(path4) as new:
    for line in new:
        count+=1
        print count
        item = line.replace('\n','').split('\t')[1]
        blob = TextBlob(re.sub('[(){}\[\]&#\'0123456789;\n]', '', item))
        for var in blob.tags:
            if var[1]=='NN' and var[0]!='ucontent':
                noun.append(var[0])
m=co.Counter(noun)
with open(path4,'w') as rank:
    for i in m.keys():
        if m[i]>1000:
            rank.write(i+'\t'+str(m[i])+'\n')         





      

        
        