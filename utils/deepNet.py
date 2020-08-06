## now implemented on colab

import numpy as np
import sys
import sklearn 
# import pyts
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import utils.prep as prep

from scipy.interpolate import interp1d
from scipy.io import loadmat

from builtins import print

import  tensorflow.compat.v1.keras as keras
# Restart runtime using 'Runtime' -> 'Restart runtime...'
# %tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)


import h5py
#from google.colab import files

# find model weights of train/test set

def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None :
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test


def save_logs(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(os.path.join(output_directory,'history.csv'), index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val,y_pred_val)
    df_metrics.to_csv(os.path.join(output_directory,'df_metrics.csv'), index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0],
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, os.path.join(output_directory,'epochs_loss.png'))

    return df_metrics

def calculate_metrics(y_true, y_pred,duration,y_true_val=None,y_pred_val=None):
    res = pd.DataFrame(data = np.zeros((1,4),dtype=np.float), index=[0],
        columns=['precision','accuracy','recall','duration'])
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_true,y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val,y_pred_val)

    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['duration'] = duration
    return res

   
def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    



def find_weights(model_dir, x_train, y_train, x_test, y_test,model_file_name = None,train_weight = False, save_weights = False):
    
    """ get weights from train/test set with a saved hdf5 model
    args: 
        x_train, y_train, x_test,y_test: np array of train, test TS (2D) and label (1D)
        model_file_name: file name of saved model
        train_weight: to get train/test weight; train_weight = True means getting weights from training set, train_weight = False means getting weights from testing set
        save_weights: whether to save the weights to txt file, default is False

    return:
        a 2D numpy array of train/test weight, shape (x.shape[0], x.shape[1])
    """
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_binary = enc.transform(y_test.reshape(-1, 1)).toarray()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], 1)
    model_dir = os.path.join(model_dir,model_file_name)
    model = keras.models.load_model(model_dir)

    # filters
    w_k_c = model.layers[-1].get_weights()[0] #  weights for each filter k for each class c

    
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]
    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)
    classes = np.unique(y_train)

    if train_weight: x,y = x_train, y_train
    else: x,y = x_test, y_test

    weights = []
    for i, ts in enumerate(x):
        ts = ts.reshape(1,-1,1)
        [conv_out, predicted] = new_feed_forward([ts])
        pred_label = np.argmax(predicted)
        orig_label = np.argmax(enc.transform([[y[i]]]))

        cas =   np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
        for k, w in enumerate(w_k_c[:, pred_label]):
            cas += w * conv_out[0,:, k]
        weights.append(cas)
    weights = np.array(weights)
    if save_weights: np.savetxt('output/'+model_file_name + "_model_weights.txt" , weights, delimiter=",")
    return weights




