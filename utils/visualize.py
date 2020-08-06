import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

import random as rd
import matplotlib
import sys
from scipy.interpolate import interp1d


# compute color for each point
def compute_color(x,maxc,minc):
	if x > 0:
		return 0.5*x/max(maxc,abs(minc))
	if x < 0:
		return 0.5*x/max(maxc,abs(minc))
	return 0.0


# compute color for each point
def compute_log_color(x):
	return (1.0/(1.0 + np.exp(-100*x)) - 0.5)*2


def compute_linewidth(x):
	return 1.0 + 4.0*x

def plot_thickness(ts,metats):
	maxw = max(metats)
	# normalize the scores
	if maxw > 0:
		metats = metats / maxw
	lwa = np.array([compute_linewidth(x) for x in metats])
	# maxc = max(metats)
	# minc = min(metats)
	colormap = np.array([compute_log_color(x) for x in metats])


	for i in range(0,len(ts)-1):
		lw = (lwa[i] + lwa[i+1])/2
		color = (colormap[i]+colormap[i+1])/2
		# plt.plot([i,i+1],ts[i:(i+2)],linewidth = lw,c=[0.5 + color,0.5 - color,0.5 - abs(color)])
		# plt.plot([i,i+1],ts[i:(i+2)],linewidth = lw,c=[color*2,0,1 - abs(color)*2])
		# plt.plot([i,i+1],ts[i:(i+2)],linewidth = lw,c=[max(0,(color - 0.5) * 2),1 - 2*abs(0.5-color),max(0,(0.5 - color)*2)])
		plt.plot([i,i+1],ts[i:(i+2)],linewidth = lw,c=[color,0,max(0,0.8 - color)])



def plot_time_series_with_highlight(ts_file, scores_file, ith):

	o = abs(ith) - 1

	tss = open(ts_file,'r').readlines()
	scores = open(scores_file,'r').readlines()

	metats = np.array([float(x) for x in scores[o].strip().split(',')])
	# keep values based on the sign of the index
	if ith < 0:
		metats = -metats
	metats[metats < 0] = 0

	yts = np.array([float(x) for x in tss[o].strip().split(',')])
	#y = int(yts[0])
	ts = yts[1:] # remove label

	plot_thickness(ts,metats)
	#plt.show()

    
    
def plot_time_series_with_color(ts, label, weight, i, dataset = 'JUMP',size = 'small', save = False, title = None):
    ''' Plot time series with weight color
        args:   ts      - 2D numpy array, original time series 
                label   - 1D numpy array, labels
                weight  - 2D numpy array, weights getting from corresponding dataset (training/testing)
                i       - int           , 0-indexed time series
                dataset - str           , name of the dataset (for saving purpose only)
                size    - str           , size of the plots, default is small for thumbnail.
                save    - boolean       , if the plot should be saved, default is False
        return: plot of time series colored by weight
    '''
    assert i>=0
    ts = ts[i]
    metats = weight[i]
    cas = metats
    
    def transform(X):
        ma,mi = np.max(X), np.min(X)
        X = (X - mi)/(ma-mi)
        return X*100
    cas = transform(cas)

    max_length1, max_length2 = len(metats),10000 #
    x1 = np.linspace(0,max_length1,num = max_length1)
    x2 = np.linspace(0,max_length1,num = max_length2)
    y1 = ts
    f = interp1d(x1, y1)

    fcas = interp1d(x1, cas)
    cas = fcas(x2)

    if size == 'small': plt.figure(figsize = (15,7.5))
    else: plt.figure(figsize = (20,10))
    
    plt.scatter(x2,f(x2), c = cas, cmap = 'jet', marker='.', s= 1,vmin=0,vmax = 100 )
    if title: plt.title(title, fontdict = {'fontsize' : 20})
    else: plt.title('Mr-SEQL_' + dataset + '_Class' +str(int(label[i])) + '_test_index ' +str(i))
    cbar = plt.colorbar()
#     plt.show()
    if save: plt.savefig('test')
#     if save: plt.savefig('Mr-SEQL_' + dataset + '_class' + str(int(label[o])) + '_testIndex' + str(o) + '_out.png',bbox_inches='tight',dpi=300)