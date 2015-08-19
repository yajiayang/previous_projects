#==============================================================================
# DM-pulse
#==============================================================================
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
from sklearn import metrics
import operator
import csv
import re
import math
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import hankel        
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy import random
import pandas as pd
from bokeh.charts import show, output_file
from bokeh.plotting import figure, show, output_file, ColumnDataSource,save
from bokeh.models import HoverTool
from collections import OrderedDict
from holtwinters import additive
from holtwinters import linear
from holtwinters import multiplicative
import statsmodels.api as sm
import statsmodels
import time
import itertools
import os

#==============================================================================
# (date,search intensity) = parse_csv(path of the csv file)
#==============================================================================
def parse_csv(path):    
    term = path.split('/')[-1].split('.')[0]
    trend = []
    index = []
    with open(path,'rb') as new:
        newread = csv.reader(new,delimiter ='\n')
        for i,var in enumerate(newread):
            if re.findall(r'\d+-\d+-\d+',str(var)) != []:
                trend.append(var[0].split(',')[1])
                index.append(var[0].split(',')[0])
    if trend == []:
        return 
    my_trend = [float(var) for var in trend[:-1]]
    index_name_tem = [var for var in index[:-1]]
    start  = index_name_tem[0].split(' - ')[0]
    end = index_name_tem[-1].split(' - ')[-1]
    index_name = pd.date_range(start,end,freq = 'W')
    return index_name,my_trend

#==============================================================================
# SARIMA : seasonal autoregressive integrated moving average
# (traingdata_date,prediction_date,traing_data,prediction_data) = sarima(steps_ahead,csv file path)
#==============================================================================
def sarima(steps,path):
    index_name,my_trend = parse_csv(path)
    dta = pd.DataFrame(my_trend)
    dta.index = index_name
    dta=dta.rename(columns = {0:'search'})
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
    return index_name,dt,my_trend,future


#==============================================================================
# HW: holt winters method(exponential smoothing)
# (traingdata_date,prediction_date,traing_data,prediction_data) = holt_pred(steps_ahead,csv file path)
#==============================================================================

def holt_pred(steps,path):
    index_name,my_trend = parse_csv(path)
    if my_trend == []:
        return 
    index_name_future = pd.date_range(index_name[-1],periods=steps+1,freq = 'W')[1:]
    future, alpha, beta, gamma, rmse = additive(my_trend[-5*52:],52,steps)
    return index_name,index_name_future,my_trend,future

#==============================================================================
# # the longest number of zeors in the list
#==============================================================================
def longest_zeros(a):
    count = 0
    current = 0
    for i,var in enumerate(a):
        if var == 0:
            j = i
            count += 1
            if count > current:
                current = count
        else:
            count = 0
    return current 

#==============================================================================
# #see if there is an abrupt change within 4 years
#==============================================================================
def abrupt_change(a):
    if np.mean(a[:-208]) <= 10 and np.mean(a[-208:]) >= 50:
        return 'Yes'
    else:
        return 'No'


#==============================================================================
# ts_vis: TimeSeries_Visualization
# input
#       steps: how many steps to predict
#       path:  path of the csv file
#       pred_type: 'hw' or 'sarima'
# output
#       html file opened in browser
#==============================================================================
def ts_vis(steps,path,pred_type):
    try:
        if pred_type == 'hw':
            index_name,index_name_future,my_trend,future = holt_pred(steps,path) 
        if pred_type == 'sarima':
            index_name,index_name_future,my_trend,future = sarima(steps,path) 
#        slope = np.polyfit(range(len(my_trend[-156:])),my_trend[-156:],1)[0]
#        if slope >= 0:
#            print 'This is a upward trending with slope: ', slope
        html_path = '/Users/royyang/Desktop/trending_project/html/'+path.split('/')[-1][:-4]+'.html'
#            html_path = 'example.html'            
        output_file(html_path, title="bohek example")
        source1 = ColumnDataSource(
                data=dict(
                    x1=index_name,
                    y1=my_trend,
                    Time1=[str(var).split()[0] for var in index_name],
                    Intensity1=my_trend 
                )
            )
            
        source2 = ColumnDataSource(
                data=dict(
                    x2=index_name_future,
                    y2=future,
                    Time1=[str(var).split()[0] for var in index_name_future],
                    Intensity1=[np.round(var,0) for var in future]
                )
            )
        
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,hover"
        
        p = figure(x_axis_type="datetime",plot_width=1000, plot_height=600, tools=TOOLS)
        
        p.line('x1','y1', color='red',legend='Past',source=source1)
        p.circle('x1','y1',size = 5,color = 'red',source=source1)
        p.line('x2','y2', color='blue', legend='Future',source=source2)
        p.circle('x2','y2',size = 8,color = 'blue',source=source2)
        p.xaxis.axis_label="Time"
        p.yaxis.axis_label="Search Intensity"
        p.title = "Search Prediction of "+path.split('/')[-1].split('.')[0]
        p.background_fill= "#cccccc"
        p.grid.grid_line_color="white"
        p.legend.label_standoff = 20
        p.legend.glyph_width = 50
        p.legend.legend_spacing = 15
        p.legend.legend_padding = 1
        p.legend.orientation = "top_left"
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ('Time', '@Time1'),
            ('Intensity', '@Intensity1'),
        ])
#            save(p)
        show(p)
    except Exception as err:
        print 'There is no content in file: '+path

#==============================================================================
# output a list of terms that has been rising recently
#==============================================================================
def rising_term():
    folder_path = '/Users/royyang/Desktop/trending_project/gtrends'
    path_list = []
    dir_list=[]
    new_file_name = []
    count = 0
    empty_list = []
    for path, subdirs, files in os.walk(folder_path):
       file_path = path 
       for filename in files:
           if re.findall('.csv',filename)!=None:
               dir_list.append(filename)
    for i,var in enumerate(dir_list):
        try:
            print i
            path = folder_path+'/'+var
            index_name,my_trend = parse_csv(path)
            slope = np.polyfit(range(len(my_trend[-104:])),my_trend[-104:],1)[0]
            if slope >= 0.2:
                path_list.append((var[:-4],slope)) 
        except Exception as err:
            count += 1
            empty_list.append(var)
    print 'Total number of empty files: ',count,empty_list  
    return path_list

#==============================================================================
# input a date and list, output the intensity of its seasonality
#==============================================================================
def seasonal_term(index_name,my_trend):
    dta = pd.DataFrame(my_trend[-156:])
    dta.index = index_name[-156:]
    dta=dta.rename(columns = {0:'search'})
    res = sm.tsa.seasonal_decompose(dta)
    seasonal = res.seasonal
    high = float(max(np.array(seasonal)))
    low = float(min(np.array(seasonal)))
    seasonal_measure = high -low 
    return seasonal_measure

#==============================================================================
# TimeSeriesVisulization automatically select best prediction algorithm
#==============================================================================
def ts_vis_auto(steps,path):
    try:
        index_name,my_trend = parse_csv(path)
        if my_trend.count(0) <= 30:
            index_name,index_name_future,my_trend,future = holt_pred(steps,path) 
        else:
            index_name,index_name_future,my_trend,future = sarima(steps,path) 
        slope = np.polyfit(range(len(my_trend[-156:])),my_trend[-156:],1)[0]
        if slope >= 0:
            print 'This is a upward trending with slope: ', slope
            html_path = '/Users/royyang/Desktop/trending_project/html/'+path.split('/')[-1][:-4]+'.html'
    #            html_path = 'example.html'            
            output_file(html_path, title="bohek example")
            source1 = ColumnDataSource(
                    data=dict(
                        x1=index_name,
                        y1=my_trend,
                        Time1=[str(var).split()[0] for var in index_name],
                        Intensity1=my_trend 
                    )
                )
                
            source2 = ColumnDataSource(
                    data=dict(
                        x2=index_name_future,
                        y2=future,
                        Time1=[str(var).split()[0] for var in index_name_future],
                        Intensity1=[np.round(var,0) for var in future]
                    )
                )
            
            TOOLS = "pan,wheel_zoom,box_zoom,reset,save,hover"
            
            p = figure(x_axis_type="datetime",plot_width=1000, plot_height=600, tools=TOOLS)
            
            p.line('x1','y1', color='red',legend='Past',source=source1)
            p.circle('x1','y1',size = 5,color = 'red',source=source1)
            p.line('x2','y2', color='blue', legend='Future',source=source2)
            p.circle('x2','y2',size = 8,color = 'blue',source=source2)
            p.xaxis.axis_label="Time"
            p.yaxis.axis_label="Search Intensity"
            p.title = "Search Prediction of "+path.split('/')[-1].split('.')[0]
            p.background_fill= "#cccccc"
            p.grid.grid_line_color="white"
            p.legend.label_standoff = 20
            p.legend.glyph_width = 50
            p.legend.legend_spacing = 15
            p.legend.legend_padding = 1
            p.legend.orientation = "top_left"
            hover = p.select(dict(type=HoverTool))
            hover.tooltips = OrderedDict([
                ('Time', '@Time1'),
                ('Intensity', '@Intensity1'),
            ])
    #            save(p)
            show(p)
    except Exception as err:
        print 'There is no content in file: '+path

#==============================================================================
#input: all the files in csv format 
#output 5 columns (key_words,longterm slope,52_week_average,seasonal_effect,predictions)
#==============================================================================
def result_summary():
    count = 0
    import re
    import os
    import time
    folder_path = '/Users/royyang/Desktop/trending_project/gtrends'
    dir_list=[]
    new_file_name = []
    for path, subdirs, files in os.walk(folder_path):
       file_path = path 
       for filename in files:
           if re.findall('.csv',filename)!=None:
               dir_list.append(filename)
    output_path = '/Users/royyang/Desktop/trending_project/hw_predict.txt'
    with open(output_path,'w') as output:
        for i, var in enumerate(dir_list):
            try:
                print i
                index_name,my_trend = parse_csv(folder_path+'/'+var)
                seasonal_measure = seasonal_term(index_name,my_trend)
                if my_trend.count(0) <= 30:
                    index_name,index_name_future,my_trend,future = holt_pred(52,folder_path+'/'+var) 
                else:
                    index_name,dt,my_trend,future = sarima(52,folder_path+'/'+var)
                slope = np.polyfit(range(len(my_trend[-156:])),my_trend[-156:],1)[0]
#                if slope >= 0:
                output.write(var[:-4]+'\t'+str(format(slope, 'f'))+'\t'+str(np.mean(my_trend[-52:]))+'\t'+str(seasonal_measure)+'\t'+','.join([str(var) for var in future])+'\n')
#                    output.write(str({var[:-4]:future})+'\n')
            except Exception as err:
                count += 1
                print 'No contents in file: ',var
    print count


#==============================================================================
# # users can specify how many weeks s/he wants to look at, future is the predicted value
#==============================================================================
def short_term_trending(weeks,future):
    low = np.inf
    high = 0
    index_high = 0
    index_low = 0
    for i,var in enumerate([float(k) for k in future.split(',')][:weeks]):
        if var > high:
            high = var
            index_high = i
    if index_high != 0:
        for i,var in enumerate([float(k) for k in future.split(',')][:index_high]):
            if var < low:
                low = var
                index_low = i
    else:
        low,index_low = high,index_high
    if low == 0:
        return (high,index_high,low,index_low,None)
    else:
        return (high,index_high,low,index_low,int((high-low)/low*100))
        


def edit_cal():
    tem_list = []
    with open('/Users/royyang/Desktop/trending_project/hw_predict.txt') as new:
        start = '2015-08-02'
        index_name_future = pd.date_range(start,periods=53,freq = 'W')
        for line in new:
            item = line.replace('\n','').split('\t')
            future = [float(var) for var in item[4].split(',')]
            avg = np.mean(future)
            index, value = max(enumerate(future), key=operator.itemgetter(1))
            index_min, value_min = min(enumerate(future), key=operator.itemgetter(1))
#            print item[0],index+1,value,str(index_name_future[index]).split()[0]
            tem_list.append({'seasonal':float(item[3]),'Current Value':float(item[2]),'Valley Value':value_min,'52_week_avg':avg,'Search Phrases':item[0],'slope':float(item[1])*100,'weeks':index+1,'Peak Value':value,'Starting Date':str(index_name_future[index]).split()[0]})
    # the group by in python has to be perform on sorted key
    tem_list = sorted(tem_list, key=lambda res: res['weeks'],reverse=False)
    with open('/Users/royyang/Desktop/trending_project/edit_cal_seasonal.csv','w') as new:
        for key, group in itertools.groupby(tem_list, lambda item: item["weeks"]):
#            test_1 = item['Peak Value']
            a = [[item["Search Phrases"],item["slope"],np.round(((item['Peak Value']-item['Valley Value'])-item['52_week_avg'])/(item['52_week_avg']+1)*100,2),item['seasonal']] for item in group]
            a_new = sorted(a, key=lambda res: res[3],reverse=True)
            new.write(str(key)+','+str(index_name_future[key-1]).split()[0]+','+','.join([item[0]+'('+str(np.round(float(item[3]),1))+')' for item in a_new])+'\n')
    with open('/Users/royyang/Desktop/trending_project/edit_cal_long_term.csv','w') as new:
        long_term_list = []
        for key, group in itertools.groupby(tem_list, lambda item: item["weeks"]):
            a = [[item["Search Phrases"],item["slope"],np.round((item['Peak Value']-item['52_week_avg'])/(item['52_week_avg']+1)*100,2)] for item in group]
            a_new = sorted(a, key=lambda res: res[1],reverse=True)
            new.write(str(key)+','+str(index_name_future[key-1]).split()[0]+','+','.join([item[0]+'('+str(item[1])+')' for item in a_new])+'\n')
    
            
#holt_pred(52,path)
#ts_vis(52,path)
#if __name__ == "__main__":
#    import re
#    import os
#    import time
#    folder_path = '/Users/royyang/Downloads/gtrends 3'
#    dir_list=[]
#    new_file_name = []
#    for path, subdirs, files in os.walk(folder_path):
#       file_path = path 
#       for filename in files:
#           if re.findall('.csv',filename)!=None:
#               dir_list.append(filename)
#    for i, var in enumerate(dir_list[20:50]):
#        time.sleep(2)
#        path = folder_path+'/'+var
#        ts_vis(52,path,'hw')
#        holt_pred(52,path)
    #        print i
    #        a = time.time()

#==============================================================================
# rising term list
#==============================================================================
rising_term_list = rising_term()
with open('/Users/royyang/Desktop/trending_project/edit_cal_recent_rising.txt','w') as new:
    for i,var in enumerate(rising_term_list):
        if i % 11 == 10:
            new.write(var+'\n')
        else:
            new.write(var+',')
            
#==============================================================================
# output prediction results for the next 52 weeks
#==============================================================================
result_summary()


#==============================================================================
# output the editorial calendar(both long term and seasonal)
#==============================================================================
edit_cal()



#==============================================================================
# sort the prediction results
#==============================================================================
res = []
output_path = '/Users/royyang/Desktop/trending_project/hw_predict.txt'
with open(output_path) as new:
    for line in new:
        item = line.replace('\n','').split('\t')
        res.append((item[0],item[1],item[2]))
        
final_list = sorted(res, key=lambda res: res[1],reverse=True)

# store the results
with open('/Users/royyang/Desktop/trending_project/hw_predict_sorted.txt','w') as new:
    for var in final_list: 
        new.write(var[0]+'\t'+var[1]+'\t'+var[2]+'\n')



#==============================================================================
# #access the stored results, output the sorted short term change
#==============================================================================
temp_res = []
weeks = 12
with open('/Users/royyang/Desktop/trending_project/hw_predict_sorted.txt') as new:
    for line in new:
        item = line.replace('\n','').split('\t')
        temp_res.append((item[0],)+short_term_trending(weeks,item[2]))

temp_res = sorted(temp_res, key=lambda temp_res: temp_res[5],reverse=True)

with open('/Users/royyang/Desktop/trending_project/'+str(weeks)+'weeks_prediction.txt','w') as old:
    old.write('keywords'+'\t'+'peak'+'\t'+'# of weeks from now(peak)'+'\t'+'valley'+'\t'+'# of weeks from now(valley)'+'\t'+'percentage increase'+'\n')
    for var in temp_res:
        old.write('\t'.join([str(k) for k in var])+'\n')        

#==============================================================================
# # visulize the results (batch)
#==============================================================================
final_list = temp_res[200:250]
for i,line in enumerate(final_list):
    if i <= 10:
#        time.sleep(2)
        a = time.time()
        path = '/Users/royyang/Desktop/trending_project/gtrends/'+line[0]+'.csv'
        ts_vis_auto(52,path)
        print time.time()-a

#==============================================================================
# visulize the resutls (one data point)
#==============================================================================
#Second_Weddings,RSS_Feeds,Popular_Websites
path = '/Users/royyang/Desktop/trending_project/gtrends/Chili_Recipes.csv'
ts_vis(52,path,'sarima')
ts_vis_auto(52,path)
    
        







#==============================================================================
# # store all the html files
#==============================================================================
final_list = []
with open('/Users/royyang/Desktop/trending_project/hw_predict.txt') as new:
    for line in new:
        final_list.append(line.replace('\n','').split('\t')[0])
        
for i,line in enumerate(final_list):
    path = '/Users/royyang/Desktop/trending_project/gtrends/'+line+'.csv'
    ts_vis(52,path,'hw')
    print i
    
    












