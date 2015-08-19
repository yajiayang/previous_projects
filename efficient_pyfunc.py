#alway use generator/iterator as return in stead of list
def load_cities_gen(path):
    with open(path) as handle:
        for line in handle:
            city,count=line.split('\t')
            yield city, int(count)
            
result=load_cities_gen('pop.tsv')
print next(result)
print next(result)
print next(result)


sum(x for _, x in pop)

# iterable container
class LoadCities(object):
    def _init_(self,path):
        self.path=path
    def _iter_(self):
        with open(self.path) as handle:
            for line in handle:
                city, count = line.split('\t')
                yield city, int(count)
                
# how to write a defensive function
def normalize_defensive(pop):
    if iter(pop) is iter(pop):
        raise TypeError(
               'Must be a container')
    total=sum(x for _, x in pop)
    for city,count in pop:
        percent=100*count/total
        yield city, percent
       
def log(message,*values):
    if not values:
        print message
    else:
        out=', '.join(str(x) for x in values)
        print ('%s:%s' %(message,out))


#alksdjflas
#asfdfsa