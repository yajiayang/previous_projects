#http://nbviewer.ipython.org/url/www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_demo.ipynb
#this script has to run in ipython (not in spyder)
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 



caffe_root = '/Users/royyang/Documents/caffe/python' 
import sys
sys.path.insert(0, caffe_root)
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

mean_filename='/Users/royyang/Documents/caffe/models/cnn_age_gender_models/mean.binaryproto'
proto_data = open(mean_filename,"rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


age_net_pretrained='/Users/royyang/Documents/caffe/models/cnn_age_gender_models/age_net.caffemodel'
age_net_model_file='/Users/royyang/Documents/caffe/models/cnn_age_gender_models/deploy_age.prototxt'
# openCV read image by BGR(0,1,2), skimage read image by RGB(2,1,0)
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

gender_net_pretrained='/Users/royyang/Documents/caffe/data/roy/bvlc_reference_caffenet_iter_45.caffemodel'
gender_net_model_file='/Users/royyang/Documents/caffe/data/roy/bvlc_reference_caffenet/deploy.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

example_image = '/Users/royyang/Desktop/googlenettxt/val/val_woman4.jpg'
input_image = caffe.io.load_image(example_image)
# the image doesn't show
_ = plt.imshow(input_image)

prediction = age_net.predict([input_image]) 

print 'predicted age:', age_list[prediction[0].argmax()]


prediction = gender_net.predict([input_image]) 

print 'predicted gender:', gender_list[prediction[0].argmax()]

#caffe.reset_all()


