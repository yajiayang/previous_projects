# Json 
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import collections as co   
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import scipy.spatial.distance as ssd
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english") 



#path = '/Users/royyang/Desktop/trending_project/keywords/Christian_Homeschooling.json'
#with open(path) as json_file:
#    json_data = json.load(json_file)
#    for var in json_data:
#        print var
#def top_related(number,path):
#    res,title = [],[]
#    with open(path) as new:
#        json_data = json.load(new)
#        for line in json_data:
##            item = line.replace('\n','').split(';')[-1].split(',')
##            other_item = line.replace('\n','').split(';')[:-1]
##            res.append([float(var) for var in item])
#            title.append([str(re.sub(r'[^\x00-\x7F]+',' ', line['keyword'])),line['search volume']])
#    city_list = []
#    path1 = '/Users/royyang/Desktop/trending_project/cities_name.txt'
#    with open(path1) as new:
#        for line in new:
#            tem = line.replace('\"','').split()
#            if len(tem) > 1:
#                city_list.append(' '.join(tem[1:]))
#    cachedStopWords = stopwords.words("english")
#    counter = co.Counter()
#    for var in title:
#        for item in [str(stemmer.stem(k)) for k in var[0].split() if not k in cachedStopWords+'\t'.join(city_list).lower().split('\t')]:
#            counter.update(co.Counter({item:int(var[1])}))
#    return [var[0] for var in title[:number]],counter.most_common(number)

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


path = final_path
def top_related(path):
    res,title = [],[]
    with open(path) as new:
        json_data = json.load(new)
        for i,line in enumerate(json_data):
            if i == 0:
                title_line = line
            title.append(str(re.sub(r'[^\x00-\x7F]+',' ', line['keyword'])))
                
    #            item = line.replace('\n','').split(';')[-1].split(',')
    #            other_item = line.replace('\n','').split(';')[:-1]
    #            res.append([float(var) for var in item])
    #        title.append([str(re.sub(r'[^\x00-\x7F]+',' ', line['keyword'])),line['search volume']])
    #title
    #len(title)
    new_title = []
    for var in title:
        new_title.append([str(stemmer.stem(item)) for item in var.lower().split()])
    bag = len(new_title)
    new_title = [' '.join(var) for var in new_title]
    df = np.array(fn_tdm_df(new_title).transpose())
    key_list = []
    while bag >= 1:
        bag,key_list = find_key_search(bag,key_list,title,df)
    return key_list,title_line
    
    

def find_key_search(bag,key_list,title,df):
    count = 0
    rem = []
    for i in range(bag):
        score = ssd.cosine(df[bag[0]],df[i])
        if score < 0.5:
            rem.append(i)
#            print title[i],i
            count +=1
    key_list.append(title[bag[0]])
#    print count
    for var in rem:
        bag.remove(var)
    return bag,key_list

    

#from collections import Counter
#Counter(range(151)) - Counter([1,2,3,4])


#==============================================================================
# output format: top_related-search_phrases: ~50 phrases
#==============================================================================
if __name__=='__main__':
    folder_path = '/Users/royyang/Desktop/trending_project/keywords'
    output_path = '/Users/royyang/Desktop/trending_project/top_related_trending.json'
    with open(output_path,'w') as new:
        for path, subdirs, files in os.walk(folder_path):
            for i,var in enumerate(files):
#                try:
                print i
                final_path = path+'/'+var
                key_list,title_line = top_related(final_path)
                new.write("{}\n".format(json.dumps({
                "phrase":title_line,
                "top_related_search": key_list
                })))
                    
#                    new.write('top_50_related-search_phrases:'+'\t'+','.join(a)+'\t'+'top_50_related-search_words:'+'\t'+','.join([var[0] for var in b])+'\n')
#                        print 'top_50_related-search_phrases:',a
#                        print 'top_50_related-search_words:',[var[0] for var in b]
#                except Exception as err:
#                    print 'No content in .json file:', final_path
  
# test 1 file      
#path =  '/Users/royyang/Desktop/trending_project/keywords/2_Way_Radios.json'
#path =  '/Users/royyang/Desktop/trending_project/keywords/Adoption_Laws.json'
#
#def see_all_words(path):
#    res,title = [],[]
#    with open(path) as new:
#        json_data = json.load(new)
#        for line in json_data:
#            title.append(str(re.sub(r'[^\x00-\x7F]+',' ', line['keyword'])))
#    return title, len(title)
#
#see_all_words(path)
#[top_related(path),len(top_related(path))]
    
        



