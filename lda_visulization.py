#heatmap
from bokeh.charts import HeatMap, output_file, show
from bokeh.palettes import YlOrRd9 as palette
from bokeh.sampledata.unemployment1948 import data
import collections as co
import pandas as pd



# store topic with probability

# construct corpus
index_all = []
with open(folder_name+'topic_with_prob.txt') as new:
    for line in new:
        item = line.strip('\n').split('\t')
        index_one = [var.split('*')[1].strip( ) for var in item[1].split('+')[:5]]
        index_all += index_one
corpus = dict(co.Counter(index_all))
corpus = dict.fromkeys(corpus, 0)

# construct value matrix
value_mt = []
with open(folder_name+'topic_with_prob.txt') as new:
    for line in new:
        corpus = dict(co.Counter(index_all))
        corpus = dict.fromkeys(corpus, 0)
        item = line.strip('\n').split('\t')
        index_one = [var.split('*')[1].strip( ) for var in item[1].split('+')[:5]]
        value_one = [var.split('*')[0].strip( ) for var in item[1].split('+')[:5]]
        corpus_one = corpus
        for i,var in enumerate(index_one):
            corpus_one[var] = float(value_one[i])
        value_mt.append([int(1000*var) for var in corpus_one.values()])

df = pd.DataFrame(value_mt)
topic_words = pd.DataFrame(corpus.keys())
df2 = df.transpose().set_index(topic_words[topic_words.columns[0]])
df3 = df2.transpose()
topic_label = pd.DataFrame(['topic_'+str(i) for i in range(topic_number)])
df3 = df3.set_index(topic_label[topic_label.columns[0]])

html_path = folder_name+'Weight_Management.html'
output_file(html_path, title="Weight_Management topics")

palette = palette[::-1]  # Reverse the color order so dark red is highest unemployment
hm = HeatMap(df3, title="Weight_Management Topics", tools = "reset,resize",
             width=1300,height=700, palette=palette)

show(hm)












