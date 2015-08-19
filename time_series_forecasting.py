##Time series forecasting
#
##use python to control R

import rpy2.robjects as robjects
robjects.r('''
        coffee <- read.csv("~/Desktop/time_series_forecasting/csv_files/correlate-coffee.csv", comment.char="#")
        a=coffee$coffee
        write.table(data.frame(a),file="~/Desktop/time_series_forecasting/csv_files/correlate-coffee_2.txt",)
        ''')


text='''
        coffee <- read.csv("~/Desktop/time_series_forecasting/csv_files/correlate-coffee.csv", comment.char="#")
        a=coffee$coffee
        write.table(data.frame(a),file="~/Desktop/time_series_forecasting/csv_files/correlate-coffee_2.txt",)
    '''
robjects.r(text)


#input data from csv file
result=[]
with open('/Users/royyang/Desktop/time_series_forecasting/csv_files/correlate-coffee_1.txt') as f:
    next(f)
    for line in f:
        item=line.replace('\n','').split(' ')
        result.append(float(item[1]))
print result

#how many data points do we have?
len(result) 

for i,item in enumerate(result):
    if i % 52 == 0:
        print i

# use 10% to test

# feature # of training set
fre=52

#form hankel matrix
from scipy.linalg import hankel        
X=hankel(result[0:-fre], result[-1-fre:-1])
y=result[fre:]

#use n datapoints
n=430

#split data into training and testing
Xtrain=X[:n]
ytrain=result[:n]
Xtest=X[n:]
ytest=y[n:]

#check if the split is correct
len(X)
len(y)
len(X[0])


from sklearn import svm
#deal with seasonality




# linear kernal
svr_lin1 = svm.SVR(kernel='linear', C=1)
y_lin1 = svr_lin1.fit(Xtrain, ytrain).predict(Xtest)
svr_lin2 = svm.SVR(kernel='linear', C=10)
y_lin2 = svr_lin2.fit(Xtrain, ytrain).predict(Xtest)
svr_lin3 = svm.SVR(kernel='linear', C=100)
y_lin3 = svr_lin3.fit(Xtrain, ytrain).predict(Xtest)
#plot results
t=range(0,len(ytest))
import matplotlib.pyplot as plt
plt.close()
plt.plot(t,y_lin1,'r--',t,ytest,'b--')
plt.plot(t,y_lin2,'g--',t,ytest,'b--')
plt.plot(t,y_lin3,'y^-',t,ytest,'b--')

# poly kernal
svr_lin1 = svm.SVR(kernel='poly', C=0.5, degree = 1)
y_lin1 = svr_lin1.fit(Xtrain, ytrain).predict(Xtest)
svr_lin2 = svm.SVR(kernel='poly', C=1, degree = 1)
y_lin2 = svr_lin2.fit(Xtrain, ytrain).predict(Xtest)
svr_lin3 = svm.SVR(kernel='poly', C=1.5, degree = 1)
y_lin3 = svr_lin3.fit(Xtrain, ytrain).predict(Xtest)


#import results from holtwinters
result_holt=[]
with open('/Users/royyang/Desktop/time_series_forecasting/csv_files/predict_coffee.txt') as f:
    next(f)
    for line in f:
        item=line.replace('\n','').split(' ')
        result_holt.append(float(item[1]))
print result_holt
len(result_holt)




plt.plot(t,result_holt,'r--',t,ytest,'b^-')
plt.ylabel('some numbers')
plt.xlabel('some numbers')
plt.show()

#set up metrics to evaluate models
from sklearn import metrics
y_true = ytest
y_pred = y_lin3

metrics_result = {'MAE':metrics.mean_absolute_error(y_true, y_pred),'MSE':metrics.mean_squared_error(y_true, y_pred)}	
print metrics_result
















