#==============================================================================
# #python multiprocessing
# 
#==============================================================================

#==============================================================================
# the total number of cpu used for multiprocessing task should be 
# n-1 where n is the total number of cores in my mac

#tasks should be roughly equal size
#map() will block until job complete, can use  map_async() to return immediately
#multiple args will need to be combined into a single list, unwrap with *

# the difference between map() and map_async()
# map() block unitl completion, map_async() can output one by one
#==============================================================================


#example0_wordcount on large txt file


#==============================================================================
#Step 1 
#use command line to split the .txt file
# split -l 10000 dump.txt new
# the line above split dump.txt into small txt file with 10000 lines in each
#==============================================================================


#==============================================================================
#step 2 
#get all the file names in a dir with new as starting file name
#==============================================================================
import os
import re

dir_list=[]
new_file_name = []
for path, subdirs, files in os.walk(r'/Users/royyang/Desktop/test'):
   file_path = path 
   for filename in files:
       if re.match('new',filename)!=None:
           #dir_list.append(os.path.join(path, filename))
           dir_list.append(filename)

 
#==============================================================================
#step 3 
#Pool Class Usage
#==============================================================================
import multiprocessing as mp
import time

#functions to be used
def counter_txt((file_path, filename)):
    with open(os.path.join(file_path, filename)) as handle:
        with open(os.path.join(file_path, 'test', filename+'.txt'),'w') as new_folder:
            for line in handle:
                item=line.replace('\n','').split('\t')
                count=len(item[1].split()+item[2].split()+item[3].split()+item[4].split())
                if count<=300:
                    new_folder.write(item[0]+'\t'+str(count)+'\n')

#create a folder if necessary 
newpath = os.path.join(file_path, 'test')
if not os.path.exists(newpath): os.makedirs(newpath)

#multiprocessing
t4 = time.time()
results=mp.Pool(processes=6).map(counter_txt, [(file_path,i) for i in dir_list])
print 'process test took' ,time.time()-t4

#merge the files and dump all the file to final_file.txt
with open(os.path.join(file_path,'final_file.txt'), 'w') as outfile:
    for fname in [(os.path.join(file_path, 'test', filename+'.txt')) for filename in dir_list]:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
#--------------------------------------End of Example0--------------------#



# example1_basic
t4 = time.time()
def cube(x):
    time.sleep(1)
    return x**3
pool = mp.Pool(processes=1)
results = pool.map(cube, range(1,7))
#print results
print 'process test took' ,time.time()-t4


#example2_results all at once
pool = mp.Pool(processes=1)
start_time = time.time()
results = [(i, pool.map(square, [i])) for i in xrange(17, 25)]
for i, result in results:
    print "Result (%d): %s (%.2f secs)" % (i, result, time.time() - start_time)

#example3_results one by one
pool = mp.Pool(processes=1)
start_time = time.time()
results = [(i, pool.map_async(square, [i])) for i in xrange(17, 25)]
for i, result in results:
    print "Result (%d): %s (%.2f secs)" % (i, result.get(), time.time() - start_time)



