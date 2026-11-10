import sys, os
sys.path.insert(0,'/nvme0n1-disk/code/mojo-jiwer/python')
os.environ['MOJO_JIWER_LIB']=sys.argv[1]
import numpy as np
from mojo_jiwer import _native
print(sys.argv[1], _native.distance([1,2,3],[1,9,3]), _native.distances([[1,2,3],[4,5]],[ [1,9,3],[4,6]]))
