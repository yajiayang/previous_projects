#==============================================================================
# #how to call python script from command line
#==============================================================================

#test if import function is ok
import sys
from sklearn import svm
from scipy.linalg import hankel        
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import numpy as np

# input the following in command line
# $ python py_commandline.py 234 23 23456
n = int(sys.argv[1])
h = int(sys.argv[2])
k = int(sys.argv[3])
print n+h+k
