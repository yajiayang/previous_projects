import collections as co
from top_search_scraper import Belieber
import re

def coffee_rel(path,term):
    with open(path) as handle:
        for line in handle:
            item = line.replace('\n','').split('\t')
            a = item[1]+item[2]+item[3]+item[4]
            match = re.findall(term,a)
            if match != []:
                yield item[0],len(match)

term = raw_input("Enter Term or enter q to quit: ")
path = '/Users/royyang/Desktop/coffee/dump.txt'
path3 = '/Users/royyang/Desktop/coffee/cof_rel_fullcontent.txt'
a=coffee_rel(path,term)
count=0
with open(path3,'w') as new:
    for item in a:
        count+=1
        new.write(item[0]+'\t'+str(item[1])+'\n')
        
r = Belieber()
r.ask([term])
result = []
with open('/Users/royyang/Desktop/Top_search_term/top_search_'+term+'.txt') as new:
    for line in new:
        item = line.replace('\n','').split('\t')
        result.append(item[1])
        
m = co.Counter(result)
# suggested item to work on

print term, count, m







