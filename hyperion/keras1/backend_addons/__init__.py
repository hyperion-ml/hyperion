from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys


from keras import backend as K

# import backend
if K.backend() == 'theano':
    sys.stderr.write('Using Theano backend addons.\n')
    from .theano_backend import *
elif K.backend() == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend addons.\n')
    from .tensorflow_backend import *
else:
    raise ValueError('Unknown backend: ' + str(K.backend()))


    
    
