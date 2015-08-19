# rename_images, prepare train_val txt

path = '/Users/royyang/Desktop/training_images'

import os
import re

dir_list=[]
new_file_name = []
for path, subdirs, files in os.walk(r'/Users/royyang/Desktop/googlenettxt/val'):
   file_path = path 
   for filename in files:
       dir_list.append(filename)
#           dir_list.append(filename)
       
       
with open('/Users/royyang/Desktop/googlenettxt/val.txt','w') as new:
    for var in dir_list:
        print var
        if '.DS_Store' not in var:
            if 'woman' in var:
                new.write(var+' '+str(0)+'\n')
            else:
                new.write(var+' '+str(1)+'\n')
                
                
#for name in /Users/royyang/Desktop/googlenettxt/train/*;do convert -resize 256x256\! $name $name; done
