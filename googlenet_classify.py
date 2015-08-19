import sys
sys.path.insert(0,'/Users/royyang/Documents/caffe/python')

import caffe
import cStringIO as StringIO
import urllib
import os
import numpy as np
import pandas as pd
import cPickle

caffe.set_mode_cpu() # set the mode either cpu or gpu

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
model_def_file = '/Users/royyang/Documents/caffe/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)
pretrained_model_file = '/Users/royyang/Documents/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)
mean_file = '/Users/royyang/Documents/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)
class_labels_file = '/Users/royyang/Documents/caffe/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)
bet_file = '/Users/royyang/Documents/caffe/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)
image_dim = 256
raw_scale = 255

def googlenet_classify(url):
    # Fetch the image from the 'url'
    string_buffer = StringIO.StringIO(urllib.urlopen(url).read())
    image = caffe.io.load_image(string_buffer)

    leafNode_result, bet_result = classify_image(image)
    return (leafNode_result, bet_result)

def classify_image(image):
    net = caffe.Classifier(model_def_file, pretrained_model_file,
                           image_dims=(image_dim, image_dim), raw_scale=raw_scale,
                           mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2,1,0))
    with open(class_labels_file) as f:
        labels_df = pd.DataFrame([
            {
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            }
            for l in f.readlines()
        ])
        labels = labels_df.sort('synset_id')['name'].values

        bet = cPickle.load(open(bet_file))
        bet['infogain'] -= np.array(bet['preferences']) * 0.1

        scores = net.predict([image], oversample=True).flatten()
            

        indices = (-scores).argsort()[:5]
        predictions = labels[indices]

        
        meta = [
            (p, '%.5f' % scores[i])
            for i, p in zip(indices, predictions)
        ]
            
            # Compute expected information gain
        expected_infogain = np.dot(
            bet['probmat'], scores[bet['idmapping']])
        expected_infogain *= bet['infogain']

            # sort the scores
        infogain_sort = expected_infogain.argsort()[::-1]
        bet_result = [(bet['words'][v], '%.5f' % expected_infogain[v])
                        for v in infogain_sort[:5]]

        return (meta, bet_result)



if __name__=='__main__':
    url = 'http://st.depositphotos.com/1044737/3472/i/950/depositphotos_34729283-Fresh-berry-fruits-background.jpg'
    res1, res2 = googlenet_classify(url)
