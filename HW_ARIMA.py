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
from sklearn import metrics
from holtwinters import additive
from holtwinters import multiplicative
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

def sarima_test(steps,path):
    index_name,my_trend = parse_csv(path)
    dta = pd.DataFrame(my_trend)
    dta.index = index_name
    dta=dta.rename(columns = {0:'search'})
    r_df = com.convert_to_r_dataframe(dta)
    y = stats.ts(r_df)
    order = R.IntVector((1,1,1))
    season = R.ListVector({'order': R.IntVector((0,1,0)), 'period' : 52})
    model = stats.arima(y[-5*52:-steps], order = order, seasonal=season)
    f = forecast.forecast(model,h=steps) 
    future = [var for var in f[3]]
    y_pred = np.array(future)
    y_true = np.array(my_trend[-steps:])
    metrics_result = {'sarima_MAE':metrics.mean_absolute_error(y_true, y_pred),'sarima_MSE':metrics.mean_squared_error(y_true, y_pred),
                  'sarima_MAPE':np.mean(np.abs((y_true - y_pred) / y_true)) * 100}	
    p1 = plt.plot(my_trend[-steps:],'*-')
    p2 = plt.plot(future)
#    p1 = plt.plot(index_name,my_trend,'r-')
#    p2 = plt.plot(index_name_future,future,'g-')
    plt.ylabel('Search Intensity')
    plt.xlabel('Year')
    plt.title('Search Prediction of '+path.split('/')[-1][:-4])
    plt.legend((p1[0], p2[0]), ["Actual","Predicted"], loc=1)
    plt.grid(True)
#    print metrics_result['sarima_MAPE']
    return metrics_result['sarima_MAPE']

def holt_pred_test(steps,path,func):
    index_name,my_trend = parse_csv(path)
    if func == 'additive':
        future, alpha, beta, gamma, rmse = additive(my_trend[-5*52:-steps],52,steps)
    if func == 'multi':
        future, alpha, beta, gamma, rmse = multiplicative(my_trend[-5*52:-steps],52,steps)
    y_pred = np.array(future)
    y_true = np.array(my_trend[-steps:])
    metrics_result = {'sarima_MAE':metrics.mean_absolute_error(y_true, y_pred),'sarima_MSE':metrics.mean_squared_error(y_true, y_pred),
                  'sarima_MAPE':np.mean(np.abs((y_true - y_pred) / y_true)) * 100}	
    p1 = plt.plot(my_trend[-steps:],'*-')
    p2 = plt.plot(future)
#    p1 = plt.plot(index_name,my_trend,'r-')
#    p2 = plt.plot(index_name_future,future,'g-')
    plt.ylabel('Search Intensity')
    plt.xlabel('Year')
    plt.title('Search Prediction of '+path.split('/')[-1][:-4])
    plt.legend((p1[0], p2[0]), ["Actual","Predicted"], loc=1)
    plt.grid(True)
#    print metrics_result['sarima_MAPE']
    return metrics_result['sarima_MAPE']


steps = 52
path = '/Users/royyang/Desktop/trending_project/gtrends/Dandruff.csv'
sarima_test(steps,path)

final_list = temp_res
sarima_win,hw_win = 0,0
for i,line in enumerate(final_list):
    path = '/Users/royyang/Desktop/trending_project/gtrends/'+line[0]+'.csv'
    sarima_error = sarima_test(steps,path)
    hw_error = holt_pred_test(steps,path)
    print 'sarima: ',sarima_error
    print 'hw: ',hw_error
    if hw_error >= sarima_error:
        sarima_win += 1
    else:
        hw_win += 1
print 'hw_win: ',hw_win,'sarima_win: ', sarima_win
        
    
    

#==============================================================================
# compute mape
#==============================================================================

steps =52
for i in range(100,150):
    try:
        i = 137
        path = '/Users/royyang/Desktop/trending_project/gtrends/'+final_list[i][0]+'.csv'
        sarima_error = sarima_test(steps,path)
        hw_error_add = holt_pred_test(steps,path,'additive')
        if sarima_error < hw_error_add:
            print i,final_list[i][0]
            print 'sarima: ',sarima_error
            print 'hw: ',hw_error_add
    except Exception as err:
        print 'error'
        
ts_vis(52,path,'sarima')
ts_vis(52,path,'hw')
ts_vis_auto(52,path)
  





















