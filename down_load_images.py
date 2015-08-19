# -*- coding: utf-8 -*-
import urllib
import time
#Latent Dirichlet Allocation for document clustering_large scale

import multiprocessing as mp
path = '/Users/royyang/Desktop/ehow_image/'
path1 = '/Users/royyang/Desktop/imageurl/url_caption_eh_part1.txt'
path2 = '/Users/royyang/Desktop/imageurl/url_caption_eh_part1_full.txt'

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

arg = spliter(800000-1,100)


result = []
with open(path1) as new:
    for i,line in enumerate(new):
        item = line.replace('\n','').split('\t')
        result.append(item[0])
        print (i)
        


def download_image((m,n)):
    error_list = []
    try:
        for var in range(m,n+1):
            urllib.urlretrieve(result[var],path+str(var)+'.jpg')
    except Exception as err:
        error_list.append(var)
    return error_list
    
    
 
res = []
start_time = time.time()
res.append(mp.Pool(processes=8).map(download_image, [(var[0],var[1]) for var in arg]))
print("--- %s seconds ---" % (time.time() - start_time))


for i, var in enumerate(res[0]):
    if i <= 400:
        if var != []:
            print (var)
type(res[0])

#error_list = []
#with open(path1) as new:
#    for i,line in enumerate(new):
#        try:
#            item = line.replace('\n','').split('\t')
#            urllib.urlretrieve(item[0],path+str(i)+'.jpg')
#        except Exception as err:
#            error_list.append(i)
#            print (i)
        
        
#with open(path1) as old:
#    with open(path2,'w') as new: 
#        for i,line in enumerate(old):
#            new.write(str(i)+'\t'+line)
