# top search-related terms by volumn
import numpy as np
import nltk
from nltk.corpus import stopwords
import collections as co   
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def top_related(number):
    path = '/Users/royyang/Desktop/trending_project/coffee.txt'
    res,title = [],[]
    with open(path) as new:
        next(new)
        for line in new:
            item = line.replace('\n','').split(';')[-1].split(',')
            other_item = line.replace('\n','').split(';')[:-1]
            res.append([float(var) for var in item])
            title.append(other_item)
    city_list = []
    path1 = '/Users/royyang/Desktop/trending_project/cities_name.txt'
    with open(path1) as new:
        for line in new:
            tem = line.replace('\"','').split()
            if len(tem) > 1:
                city_list.append(' '.join(tem[1:]))
    cachedStopWords = stopwords.words("english")
    counter = co.Counter()
    for var in title:
        for item in [str(stemmer.stem(k)) for k in var[0].split() if not k in cachedStopWords+'\t'.join(city_list).lower().split('\t')]:
            counter.update(co.Counter({item:int(var[1])}))
    return [var[0] for var in title[:number]],counter.most_common(number)
    
if __name__=='__main__':
    a,b = top_related(50)
    print 'top_50_related-search_phrases:',a
    print 'top_50_related-search_words:',[var[0] for var in b]
    





    

    









                            
