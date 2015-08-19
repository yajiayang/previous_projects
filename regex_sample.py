# install all the data in nltk
import nltk
nltk.download('all')
# metacharacter
#[.^$*+?{}[]\|]
# match the character not listed by complementing the set
[^5] # match any character except 5
# backslash can be used to precede metacharacters to remove theri special meaning
\[ or \\
# predefined special sequence
\d # match all decimal digits,eg.[0-9]
\D # match all non-digit characters, eg.[^0-9]
\s # match all whitespace charcaters, eg.[ \t\n\r\f\v]
\S # match all non-whitespace characters, eg.[^ \t\n\r\f\v]
\w # match all alphanumeric character,eg.[a-zA-Z0-9]
\W # match all non-alphanumeric characters, eg.[^a-zA-Z0-9]
# '.' is used to match any charcater except newline
# '*' repeating matches(0 or more times)
ca*t eg.(ct,cat,caat,caaat)
# '+ repeating matches(1 or more times)'
ca+t eg.(cat,caat,caaat)
# '?' repeating matches(1 or zero times)
home-?brew,eg(homebrew or home-brew)
# '{m.,n}' repeating matches (minnum m times and maximum n times)
a/{1,3}b eg(a/b,a//b,a///b)
# use raw string to handle backslash plague
'\\\\section'=r'\section'

import re
# match: determine if the RE matches at he beginning of a string
# serach: determine if the RE matches at any location of a string
p=re.compile('abc')
m=p.match('abcdefabababc')
m.group()
m.start()
m.end()
m.span()

m=re.match('abc','abcdefabababc')
m.group()

# findall: substring where RE matches, and return them as a list
p=re.compile('[ab]+c')
p.findall('abcdefabababc')

re.findall('abc','abcdefabababc')
re.findall('abc','defabab abc')



# finditer: find all substring where RE matches, and return them as a iterator
p=re.compile('[ab]+c')
m=p.finditer('abcdefabababc')
for item in m:
    print item.span()
    print item.start()
    print item.end()
    print item.group()
    
m=re.finditer('ac','abcdefabababc')

# \b word boundry
a=re.search(r'\bfamily\b', 'we are family')
a.group()

#the use of groups
re.search('(ab)+','cababababab').span()
m=re.search('(a(b)c)d','abcd')
m.group(0)
m.group(1)
m.group(2)
m.groups() #from 1 to the end

# split paragraph into sentences and match a word then output the sentence
a='''Have some desserts fun this grilling! season by trying some delicious new recipes. Your curiosity desserts, not to mention your taste buds, will surely be piqued by this collection. From appetizers to desserts, delight your guests from start to finish with things you never knew could be cooked on a grill.'''
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
results='\n-----\n'.join(tokenizer.tokenize(a))
print results

m=re.finditer('desserts',results)
for item in m:
    loc_back,loc_forw=item.start(),item.end()
    while loc_back>=0:
        if results[loc_back]=='\n':
            #print loc_back
            break
        loc_back-=1
    while loc_forw<len(results):
        if results[loc_forw]=='\n' or loc_forw+1==len(results):
            #print loc_forw
            break
        loc_forw+=1        
    print results[loc_back+1:loc_forw+1]
    

# how to calculate similarity of two strings
import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

text1 = 'a large group of fish a schooled of small glittering fish swam by'
text2 = 'TA school of fish is a collection of many fish that all travel together'
text3 = 'Starting a private school is a challenging and time-intensive project. The rewards are knowing that you are making a difference in the lives of children and families'
text4='frends the the'
text5='family the the'
text6='The children are schooled at great cost to their parents in private institutions'
vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)

cosine = get_cosine(vector1, vector2)

print 'Cosine:', cosine

# how to do pos tagging in nltk
import nltk
text=nltk.word_tokenize(text6)
text_pos=nltk.pos_tag(text)


    



    



























