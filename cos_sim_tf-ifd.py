#==============================================================================
# #calculate cosine_similarity based on tf-idf score
#==============================================================================

import re
import time
import multiprocessing as mp
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use grep in command line
# grep squash ls_similarity_pumpkin.txt
# output the lines with the searched term in it

# ------step 0 ----------------------#
# Initiation
docs = []
titles = []
path = '/Users/royyang/Desktop/coffee/cof_rel_fullcontent.txt'
path1 = '/Users/royyang/Desktop/dump.txt'

path5 = '/Users/royyang/Desktop/coffee/squash_rel_fullcontent.txt'
path6 = '/Users/royyang/Desktop/coffee/apple_rel_fullcontent.txt'
path7 = '/Users/royyang/Desktop/coffee/pumpkin_rel_fullcontent.txt'
path8 = '/Users/royyang/Desktop/coffee/pears_rel_fullcontent.txt'
path9 = '/Users/royyang/Desktop/coffee/cranberries_rel_fullcontent.txt'
# ------End of step 0 ----------------------#





# ------step 1----------------------#
with open(path7) as new:
    for line in new:
        item = line.replace('\n','').split('\t')
        docs.append(re.sub('[(){}\"\[\]&#\'_0123456789;\n]', '', item[2]))
        titles.append(item[0]) 

# spliter fucntion, input [total number of files,how many parts to split]
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

# multiprocessing to extract noun
def extract_noun_part((m,n)):
    res = []
    for i, cont in enumerate(docs[m:n]): 
        print (i)
        cont = cont.replace('\\n','').lower()
        noun = ''
        blob = TextBlob(cont)
        for var in blob.tags:
            if var[1] in ['NN'] and var[0]!='ucontent':
                noun = noun + var[0] + ' '
        res.append(noun)
    return res

arg = spliter(len(docs),10)
start_time = time.time()
result = mp.Pool(processes=7).map(extract_noun_part, [(var[0],var[1]) for var in arg])
print("--- %s seconds ---" % (time.time() - start_time))


len(result)
len(result[0])

fin_res = []
for var in result:
    for item in var:
        fin_res.append(item)
 
 
len(item)
len(fin_res)
len(fin_res[0])       
len(result)
len(result[0][1])

res = fin_res

len(res)
res[0]
type(res)


# Calculate tf-idf score for the documents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(res)
vocab = vectorizer.get_feature_names()

# multiprocessing function take 600seconds to calculate 100k articles
def cos_simi_part((m,n)):
    result = []    
    var = cosine_similarity(X[m:n],X)
    for index_a,item in enumerate(var):
        current_score = 0
        for j,each in enumerate(item):
            if each >= current_score and each < 0.999:
                current_score = each
                index_b = j
        result.append([current_score,m+index_a,index_b])
    return result
    

start_time = time.time()
result = mp.Pool(processes=7).map(cos_simi_part, [(var[0],var[1]) for var in arg])
print("--- %s seconds ---" % (time.time() - start_time))


fin_res = []
for var in result:
    for item in var:
        print (item)
        fin_res.append(item)
        
len(fin_res)

path2 = '/Users/royyang/Desktop/ls_similarity_pumpkin.txt'
with open(path2,'w') as new:
    for i,line in enumerate(fin_res):
        new.write(titles[line[1]]+'\t'+titles[line[2]]+'\n')
#        if 'squash' not in titles[line[1]] and 'squash' not in titles[line[2]]:

len(titles) 



