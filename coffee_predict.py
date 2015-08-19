#use n datapoints

    # of features of training set
#fre=50
    # how many steps to predict
#step=8
#fea=20
#cof=1
def coffee_pred(step):
    n=1290-step
    fea=20
    cof=1
    from sklearn import svm
    from scipy.linalg import hankel        
    import matplotlib.pyplot as plt
    from sklearn import metrics
    import math
    import numpy as np
    #input data from csv file
    path='/Users/royyang/Desktop/time_series_forecasting/csv_files/coffee_ls.txt'
    path1 = '/Users/royyang/Desktop/time_series_forecasting/csv_files/coffee_ls_nor.txt'
    result_tem=[]
    with open(path) as f:
        next(f)
        for line in f:
            item=line.replace('\n','').split(' ')
            result_tem.append(float(item[1]))
    mean = np.mean(result_tem)
    sd = np.std(result_tem)
    result = (result_tem-mean)/sd
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
    y_lin1 = svr_lin1.fit(Xtrain, ytrain)
    
    return int(y_lin1.predict(result[-1-fea:-1])*sd+mean)


