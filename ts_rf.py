def ts_rf(n,fea,step,ntrees,njobs):
    #Random Forest Model for time series prediction
    #from sklearn import svm
    import math
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from scipy.linalg import hankel        
    import numpy as np
    from sklearn.ensemble.forest import RandomForestRegressor
    #input data from csv file
    #use n datapoints
    #n=1100
    #    # of features of training set
    ##        fre=50
    #    # how many steps to predict
    #step=29
    #fea=50
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
    # random forest
    rf = RandomForestRegressor(n_estimators = ntrees, n_jobs=njobs)
    rf_pred = rf.fit(Xtrain, ytrain).predict(Xtest)
    #a = rf.transform(Xtrain,'median')
    
    #plot results
    LABELS = [x[-6:] for x in date[n+fea+step-1:n+fea+step-1+len(ytest)]]    
    t=range(n,n+len(ytest))
    #    plt.show()
    #    plt.plot(t,y_lin1,'r--',t,ytest,'b^-')
    #    plt.plot(t,y_lin2,'g--',t,ytest,'b^-')
    ypred = rf_pred*sd+mean
    ytest = ytest*sd+mean
    line1, = plt.plot(t,ypred,'r*-')
    plt.xticks(t, LABELS)
    line2, = plt.plot(t,ytest,'b*-')
#            plt.xlim([500,510])
    plt.legend([line1, line2], ["Predicted", "Actual"], loc=2)
        
        
        
        
        #plt.show()
        #plt.plot(xrange(n),result[0:n],'r--',t,y_lin3,'b--',t,ytest,'r--')
        
        
    y_true = ytest
    y_pred = ypred
    metrics_result = {'rf_MAE':metrics.mean_absolute_error(y_true, y_pred),'rf_MSE':metrics.mean_squared_error(y_true, y_pred),
                      'rf_MAPE':np.mean(np.abs((y_true - y_pred) / y_true)) * 100}	
    print metrics_result

