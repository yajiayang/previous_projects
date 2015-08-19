from pandas import read_csv, DataFrame, Series
import statsmodels.api as sm
import rpy2.robjects as R
from rpy2.robjects.packages import importr
import pandas.rpy.common as com
from pandas import date_range
import numpy as np
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import time
forecast = importr('forecast')
stats = importr('stats')
tseries = importr('tseries')

def parse_csv(path):    
    term = path.split('/')[-1].split('.')[0]
    trend = []
    index = []
    with open(path,'rb') as new:
        newread = csv.reader(new,delimiter ='\n')
        for i,var in enumerate(newread):
#            if re.findall(r'\d+-\d+-\d+',str(var)) != [] and int(str(var[0])[0:4])>=2007:
            if re.findall(r'\d+-\d+-\d+',str(var)) != []:
                trend.append(var[0].split(',')[1])
                index.append(var[0].split(',')[0])
    if trend == []:
        return 
    my_trend = [float(var) for var in trend[1:]]
    index_name_tem = [var for var in index[1:]]
    start  = index_name_tem[0].split(' - ')[0]
    end = index_name_tem[-1].split(' - ')[-1]
    index_name = pd.date_range(start,end,freq = 'W')
    return index_name,my_trend
    
path = '/Users/royyang/Desktop/trending_project/gtrends/Popular_Websites.csv'

def sarima(steps,path):
    index_name,my_trend = parse_csv(path)
    dta = pd.DataFrame(my_trend)
    dta.index = index_name
    dta=dta.rename(columns = {0:'search'})
    #dta.plot(figsize=(10,4))
    
    #==============================================================================
    # check stationarity
    #==============================================================================
    #r_df = com.convert_to_r_dataframe(DataFrame(dta))
    #y = stats.ts(r_df)
    #ad = tseries.adf_test(y, alternative="stationary", k=52)
    #a = ad.names[:5]
    #{ad.names[i]:ad[i][0] for i in xrange(len(a))}
    
    #==============================================================================
    # check the seasonality
    #==============================================================================
    #diff1lev = dta.diff(periods=1).dropna()
    #diff1lev.plot(figsize=(12,6))
    #diff1lev_season = diff1lev.diff(52).dropna()
    #r_df = com.convert_to_r_dataframe(DataFrame(diff1lev_season))
    #diff1lev_season1lev = diff1lev_season.diff().dropna()
    
    #==============================================================================
    # check stationarity after difference
    #==============================================================================
    #y = stats.ts(r_df)
    #ad = tseries.adf_test(y, alternative="stationary", k=52)
    #a = ad.names[:5]
    #{ad.names[i]:ad[i][0] for i in xrange(len(a))}
    
    
    #==============================================================================
    # plot acf and pacf
    #==============================================================================
    #fig = plt.figure(figsize=(12,8))
    #ax1 = fig.add_subplot(211)
    #fig = sm.graphics.tsa.plot_acf(diff1lev_season1lev.values.squeeze(), lags=150, ax=ax1)
    #ax2 = fig.add_subplot(212)
    #fig = sm.graphics.tsa.plot_pacf(diff1lev_season1lev, lags=150, ax=ax2)
    #fig
    
    r_df = com.convert_to_r_dataframe(dta)
    y = stats.ts(r_df)
    order = R.IntVector((1,1,1))
    season = R.ListVector({'order': R.IntVector((0,1,0)), 'period' : 52})
    a = time.time()
    model = stats.arima(y, order = order, seasonal=season)
    print time.time()-a
    f = forecast.forecast(model,h=steps) 
    future = [var for var in f[3]]
    dt = date_range(dta.index[-1], periods=len(future)+1,freq='W')[1:] #создаем индекс из дат
    pr = Series(future, index = dt)
#    dta.plot(figsize=(12,6))
#    pr.plot(color = 'red')
    return index_name,dt,my_trend,future
    














