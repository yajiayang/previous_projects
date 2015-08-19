#
import collections as co
import re

path = '/Users/royyang/Desktop/coffee_drinker.txt'
path1 = '/Users/royyang/Desktop/coffee_drinker_ls.txt'
count = 0
noun = []
with open(path) as new:
    next(new)
    for item in new:
        count+=1
        print count
        for var in re.sub('[\"0123456789\n]', '', item).replace(' ','').split('-'):
            if var != '':
                noun.append(var)
len(noun)        
m=co.Counter(noun)
with open(path1,'w') as rank:
    for i in m.keys():
        if m[i]>100:
            print i,m[i]
            rank.write(i+'\t'+str(m[i])+'\n') 
