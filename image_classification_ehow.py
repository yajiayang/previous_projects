caffe_root = '/Users/royyang/Documents/caffe/python'
tem_path = '/Users/royyang/Desktop/imageurl/url_caption_eh_part1.txt'
res_path = '/Users/royyang/Desktop/imageurl/deep_learning_results.txt'
import sys
sys.path.insert(0, caffe_root)
import caffe
import googlenet_classify as net
import time
import multiprocessing as mp


def image_test((m,n)):
    print m,n
    new.write('\n')
    for i in range(m,n):
        try:
            example_image = '/Users/royyang/Desktop/ehow_image/'+str(i)+'.jpg'
            input_image = caffe.io.load_image(example_image)
            leafNode_result, bet_result = net.classify_image(input_image)
            new.write(str(i)+'\t'+url[i]+'\t'+str(bet_result)+'\t'+str(leafNode_result)+'\n')
        except Exception as err:
            print i
            
def image_test_new(m):
    try:
        example_image = '/Users/royyang/Desktop/ehow_image/'+str(m)+'.jpg'
        input_image = caffe.io.load_image(example_image)
        leafNode_result, bet_result = net.classify_image(input_image)
        return (str(m)+'\t'+url[m]+'\t'+str(bet_result)+'\t'+str(leafNode_result)+'\t'+str(caption[m])+'\n')
    except Exception as err:
        return m

if __name__=='__main__':
    url = []
    caption = []
    with open(tem_path) as new:
        for line in new:
            item = line.replace('\n','').split('\t')
            url.append(item[0])
            caption.append(item[1])
    start_time = time.time()
##    for i in range(10):
##        image_test_new(i)
    m = 540000
    n = 800000
    res = mp.Pool(processes=8).map_async(image_test_new, range(m,n))
    results = (res.get())
    err_list = []
    print("--- %s seconds ---" % (time.time() - start_time))
    with open(res_path, 'w') as new:
        for i,var in enumerate(results):
            try:
                new.write(var)
            except Exception as err:
                err_list.append(i+m)
    print err_list

##caffe.reset_all()


##    mp.Pool(processes=8).map(image_test, [(var[0],var[1]) for var in arg])
##with open(res_path,'w') as new:
##    start_time = time.time()
##    mp.Pool(processes=8).map(image_test, [(var[0],var[1]) for var in arg])
##    print("--- %s seconds ---" % (time.time() - start_time))
