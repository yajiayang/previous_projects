# redis mass insertion with python

import redis

# Connection to 0 database (default in redis)
r = redis.Redis(host="localhost",db=0)

# inserting client hashmaps
r.hmset('client:1', {'name':'John', 'company':'Microsoft'})
r.hmset('client:2', {'name':'James', 'company':'Apple'})

# inserting a list of domains for client 1
r.rpush('client:1:domains','www.microsoft.com','www.msn.com')

#to print values in stdout
print(r.hgetall('client:1'))


#insert set
with open('/Users/royyang/Desktop/data.txt') as f:
    for line in f:
        print line
        r.set(line.split()[1],line.split()[2])

print r.get('yajia7')

