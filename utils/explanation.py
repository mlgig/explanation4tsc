import numpy as np
import pandas as pd
from sklearn import metrics
import os
import sys

from utils.rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV

def all_accuracy(train_x, train_y, trained_model, classifier = 'ResNetCAM', noise_type = 1, ds = 'CMJ'):
    '''
    Calculate accuracy for all noisy data range with test data
    '''
    ms = trained_model   
    accuracy = []
    
    col_names = ['dataset', 'type', 'noise_level', 'acc']
    df = pd.DataFrame(columns = col_names)

    for noise_level in range(0,101,10):
        # load test data
        test_file  = 'output/%s_%s%d_type%d_TEST.txt' %(ds, classifier, noise_level, noise_type)

        test_data = np.genfromtxt(test_file, delimiter=',')
        test_x, test_y = test_data[:,1:], test_data[:,0]
        predicted = ms.predict(test_x)
        
        acc = metrics.accuracy_score(test_y, predicted)
        df = df.append({'dataset': ds, 'type': noise_type, 'noise_level': noise_level,  'acc': acc }, 
                   ignore_index=True)
        accuracy.append(acc)
    return df, np.array(accuracy)

