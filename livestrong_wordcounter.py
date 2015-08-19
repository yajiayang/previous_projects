#==============================================================================
# count the number of words for each article in livestrong, output a list of articles with less
# than 300 words with asending order
#==============================================================================

# the command to sort a txt file by column
# cat dump1.txt | sort -n -k2 > livestrong_thin.txt

import pymongo
from pymongo import MongoClient# Comes with pymongo
import re
import unicodedata
#how to handle unicode in python
#re.sub(r'[^\x00-\x7F]+',' ', i['description']

mongo = MongoClient('cmereporting.prod.dm.local')
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



path='/Users/royyang/Desktop/dump/dump.txt'
path1='/Users/royyang/Desktop/dump/dump1.txt'

count=0
with open(path,'w') as new:
       for i in b:
           print count
           count+=1
           new.write(i['_id']+'\t'+(' '.join(re.sub(r'[^\x00-\x7F]+',' ', i['description']).split()) if ('description' in i) else ' ')+'\t'+
           (' '.join(str(i['sections']).split()) if ('sections' in i) else ' ')+'\t'+
           (' '.join(str(i['tips']).split()) if ('tips' in i) else ' ')+'\t'+
           (' '.join(str(i['warnings']).split()) if ('warnings' in i) else ' ')+'\n')


# directly calculate # of articles with less than 300 words
def counter_txt(path):
    with open(path) as handle:
        for line in handle:
            item=line.replace('\n','').split('\t')
            count=len(item[1].split()+item[2].split()+item[3].split()+item[4].split())
            if count<=300:
                yield item[0],count      

a=counter_txt(path)

count=0
with open(path1,'w') as new:
    for item in a:
        count+=1
        print count
        new.write(item[0]+'\t'+str(item[1])+'\n')
        
        




