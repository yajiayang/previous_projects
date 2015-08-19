# Python data structure
import collections as co
a='''A search engine lists web pages on the Internet. This facilitates research by offering an immediate variety of applicable options. Possibly useful items on the results list include the source material or the electronic tools that a web site can provide, such as a dictionary, but the list itself, as a whole, can also indicate important information, perhaps inasmuch as a book can be judged by its title.
Referencing search engine results is a quick way to either present (what is notable) or delete (what is not verifiable) source material, depending on their reliability. There is a high demand for reliability on Wikipedia. Discerning the reliability of the source material is an especially core skill for using the web, while the wiki itself only facilitates the creation multiple drafts. As presentations and deletions progress, this variety of choices for input tend to produce the desired objective—a neutral viewpoint. Depending on the type of query and kind of search engine, this variety can open up to a single author.'''
WORD=re.compile('\w+')
words = WORD.findall(a)
m=co.Counter(words)
for i in m.keys():
    print i,m[i]