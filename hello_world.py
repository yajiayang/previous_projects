
import sys
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
n = int(sys.argv[1])
h = int(sys.argv[2])
k = int(sys.argv[3])
#n,h = int(n), int(h)
#n = int(input("Please enter number of datapoints: "))
#fea = int(input("Please enter number of feature: "))
#step = int(input("Please enter prediction steps: "))

from sklearn import svm
from scipy.linalg import hankel        
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import numpy as np

print n+h+k

#input data from csv file
#use n datapoints
#n=400
#    # of features of training set
#    fre=50
#    # how many steps to predict
#step=3
#fea=6
#result=[]
#with open('/Users/royyang/Desktop/time_series_forecasting/csv_files/correlate-coffee_1.txt') as f:
#    next(f)
#    for line in f:
#        item=line.replace('\n','').split(' ')
#        result.append(float(item[1]))
##form hankel matrix
#X=hankel(result[0:-fea-step+1], result[-1-fea:-1])
#y=result[fea+step-1:]
##split data into training and testing
#Xtrain=X[:n]
#ytrain=y[:n]
#Xtest=X[n:]
#ytest=y[n:]
## linear kernal
#svr_lin1 = svm.SVR(kernel='linear', C=0.01)
#y_lin1 = svr_lin1.fit(Xtrain, ytrain).predict(Xtest)
#svr_lin2 = svm.SVR(kernel='linear', C=0.1)
#y_lin2 = svr_lin2.fit(Xtrain, ytrain).predict(Xtest)
#svr_lin3 = svm.SVR(kernel='linear', C=1)
#y_lin3 = svr_lin3.fit(Xtrain, ytrain).predict(Xtest)
##plot results
#t=range(n,n+len(ytest))
#plt.show()
##    plt.plot(t,y_lin1,'r--',t,ytest,'b^-')
##    plt.plot(t,y_lin2,'g--',t,ytest,'b^-')
#line1, = plt.plot(t,y_lin3,'r*-')
#line2, = plt.plot(t,ytest,'b')
##    plt.xlim([500,510])
#plt.legend([line1, line2], ["Predicted", "Actual"], loc=2)
#
#
#
#
##    plt.show()
##    plt.plot(xrange(n),result[0:n],'r--',t,y_lin3,'b--',t,ytest,'r--')
#
#
#y_true = ytest
#y_pred = y_lin2
#
#metrics_result = {'MAE':metrics.mean_absolute_error(y_true, y_pred),'MSE':metrics.mean_squared_error(y_true, y_pred),
#                  'MAPE':np.mean(np.abs((y_true - y_pred) / y_true)) * 100}	
#print metrics_result