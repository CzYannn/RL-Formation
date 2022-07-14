import argparse
import numpy as np 
import torch as T 
import numpy as np 



def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def set_seed(seed):
    # set seed
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)

