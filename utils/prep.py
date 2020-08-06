import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

import random as rd
import matplotlib
import sys
from scipy.interpolate import interp1d
import os


def convert_ts_file(file):
    """Convert TS file format to TS + label
        args: txt file directory
        return: ts + label
        
        use: 
        
        train = 'JumpResampledNoisyType1_TRAIN.txt'
        test  = 'JumpResampledNoisyType1_TEST'
        x_train, y_train = convert_ts_file(train)
        x_test, y_test  = convert_ts_file(test)

    """
    tss = open(file,'r').readlines()
    yts = np.array([[float(x) for x in tss[o].strip().split(',')] for o in range(len(tss))])
    ts = [x[1:] for x in yts]
    try:
        label = np.array([int(x[0]) for x in yts])
    except:
        label = np.array([int(float(x[0])) for x in yts])
    print('data_file = ',file,'; nrow = ',len(ts), '; ncol = ', len(ts[0]))
    return ts, label




def convert_metats_file(file):
    """Convert metats file format to np array of weights
        args: txt file directory
        return: np array of weight
        
        use: 
        test_weight = 'test_scores'
        test_weight = convert_metats_file(test_weight)

    """
    scores = open(file,'r').readlines()
    weight = np.array([[float(x) for x in scores[o].strip().split(',')] for o in range(len(scores))])
    return weight



def write_to_std(ts,label, dataset , train = False):
    """ Write np array of TS and label to a standard format
        args: ts - numpy array of n TS
             label - label
       return: txt file in format (label, ts)
    """
    assert dataset != None
    assert len(ts) == len(label)
    
    if train: fileName = os.path.join('output' ,dataset+ '_TRAIN.txt')
    else:     fileName = os.path.join('output' ,dataset+ '_TEST.txt')
    n = len(ts)
    label = np.reshape(label, (n,-1))
    ans = np.append(label,ts, axis = 1)
    np.savetxt(fileName, ans, delimiter=",")