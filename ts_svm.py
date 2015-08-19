#
#import sys
#n = int(sys.argv[1])
#fea = int(sys.argv[2])
#step = int(sys.argv[3])
#cof = int(sys.argv[4])
def ts_svm(n, fea, step):
    from sklearn import svm
    from scipy.linalg import hankel        
    import matplotlib.pyplot as plt
    from sklearn import metrics
    import math
    import numpy as np
    #input data from csv file
    #use n datapoints
    #    n=1200
    #        # of features of training set
    #    #        fre=50
    #        # how many steps to predict
    #    step=29
    #    fea=50
    cof=1
    path='/Users/royyang/Desktop/time_series_forecasting/csv_files/coffee_ls.txt'
    path1 = '/Users/royyang/Desktop/time_series_forecasting/csv_files/coffee_ls_nor.txt'
    result_tem=[]
    date = []
    with open(path) as f:
        next(f)
        for line in f:
            item=line.replace('\n','').split(' ')
            result_tem.append(float(item[1]))
            date.append(item[2])
    
    mean = np.mean(result_tem)
    sd = np.std(result_tem)
    result=(result_tem-mean)/sd
    #form hankel matrix
    X=hankel(result[0:-fea-step+1], result[-1-fea:-1])
    y=result[fea+step-1:]
    #split data into training and testing
    Xtrain=X[:n]
    ytrain=y[:n]
    Xtest=X[n:]
    ytest=y[n:]
    # linear kernal
    svr_lin1 = svm.SVR(kernel='linear', C=cof)
    y_lin1 = svr_lin1.fit(Xtrain, ytrain).predict(Xtest)
    #plot results
    LABELS = [x[-6:] for x in date[n+fea+step-1:n+fea+step-1+len(ytest)]]    
    t=range(n,n+len(ytest))
    #    plt.show()
    #    plt.plot(t,y_lin1,'r--',t,ytest,'b^-')
    #    plt.plot(t,y_lin2,'g--',t,ytest,'b^-')
    ypred = y_lin1*sd+mean
    ytest = ytest*sd+mean
    line1, = plt.plot(t,ypred,'r*-')
    plt.xticks(t, LABELS)
    line2, = plt.plot(t,ytest,'b*-')
    #    plt.xlim([500,510])
    plt.legend([line1, line2], ["Predicted", "Actual"], loc=2)
    
     
    
    
    #plt.show()
    #plt.plot(xrange(n),result[0:n],'r--',t,y_lin3,'b--',t,ytest,'r--')
    
    
    y_true = ytest
    y_pred = ypred
    metrics_result = {'svm_MAE':metrics.mean_absolute_error(y_true, y_pred),'svm_MSE':metrics.mean_squared_error(y_true, y_pred),
                      'svm_MAPE':np.mean(np.abs((y_true - y_pred) / y_true)) * 100}	
    print metrics_result
    
    

#print X[1],y[0]

#print result[0:10]
#print Xtrain[0:5]
#print ytrain[0:5]
#tsf(1200,20,14,1)


