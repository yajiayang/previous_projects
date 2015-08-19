# Time-Series Analysis in Pandas
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy import random
pd.__version__

# cumsum() the cumulative sum
plt.plot(np.random.randn(1000).cumsum())
np.arange(5)[:2]

# index in Series
index = ['a','b','c','d','e']
s = Series(np.arange(5),index=index)
s[:3]
s['d']
s['b':]
s[[4]]
s[['a','c']]

# create date_range by day
dates = pd.date_range('2012-07-16','2012-07-21')
atemps = Series([101.4,99,90,232,233,123],index = dates)
atemps.index[2]

sdtemps = Series([73,78,77,78,78,77],index = dates)
temps = DataFrame({'Austin':atemps,'San Diego':sdtemps})
temps['diff'] = temps['San Diego'] - temps['Austin']

del temps['diff']
temps['Austin']
idx = temps.index[2]
temps.ix[[1,2,3],'Austin']

temps.mean()
#compute mean over the row
np.randn(5,5).mean(0)
#compute mean over the column
np.randn(5,5).mean(1)

















