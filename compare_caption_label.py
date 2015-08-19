caffe_root = '/Users/royyang/Documents/caffe/python' 
import sys
sys.path.insert(0, caffe_root)
import caffe

import googlenet_classify as net

def spliter(num,n_jobs):
    unit = num//n_jobs
    arg = []
    for i in range(n_jobs):
        start = i*unit
        end = (i+1)*unit
        if i==n_jobs-1:
            arg.append([start,num])
            break
        arg.append([start,end])
    return arg

arg = spliter(800000-1,100)



f = open('/Users/royyang/Desktop/url_caption.txt','r')
i=0
with open('/Users/royyang/Desktop/test.txt','w') as new:
    for line in f:
        if i <= 2:
            print i
            url = line.split('\t')[0]
            caption = line.split('\t')[1]
            leafNode_result, bet_result = net.googlenet_classify(url)
##            print leafNode_result
##            print bet_result
##            print caption
##            print url
            new.write(str(i)+'\t'+url+'\t'+str(bet_result)+'\t'+str(leafNode_result)+'\t'+str(caption)+'\n')
            i += 1



