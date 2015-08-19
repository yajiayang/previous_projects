#tsf(n, fea, step,cof)
for i in range(1,30):
    tsf(1100,20,i,1)
    

res=[]  
for i in range(1,30):
    print i
    res.append(coffee_pred(i))

import matplotlib.pyplot as plt  
path='/Users/royyang/Desktop/time_series_forecasting/csv_files/coffee_ls_date.txt'
result_tem=[]
date = []
with open(path) as f:
    next(f)
    for line in f:
        item=line.replace('\n','').split(' ')
        date.append(item[1])

label = ['2015-07-13']
for i in date:
    label.append(i)
LABELS = [x[-6:] for x in label[:30]]    
t=range(0,len(res))
line1, = plt.plot(t,res,'r*-')
#plt.xticks(t, LABELS)
plt.grid()
plt.title('Prediction of next month coffee inputs in myplate(Day0: 07/13/2015 )')

    



        
        
