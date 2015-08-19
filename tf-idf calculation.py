#tf-idf calculation
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix


a='A search engine lists web pages on the Internet. This facilitates research by offering an immediate variety of applicable options. Possibly useful items on the results list include the source material or the electronic tools that a web site can provide, such as a dictionary, but the list itself, as a whole, can also indicate important information, perhaps inasmuch as a book can be judged by its title.Referencing search engine results is a quick way to either present (what is notable)'
corpus = ["This is very strange",'This is very nice']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(res)
# convert sparse matrix to dense matrix
X_dense = X.toarray()

# just print idf score
idf = vectorizer.idf_
dict(zip(vectorizer.get_feature_names(), idf))


    
