import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

import random as rd
import matplotlib
import sys
from scipy.interpolate import interp1d
import os



class Noise:
    def __init__(self,sequence, label, train_x, weight, dataset_name = 'JumpResampled', weight_model = 'MrSEQL'):
        self.sequence = sequence
        self.label = label
        self.train_x = train_x
        
        assert self.sequence.shape[1] == self.train_x.shape[1]
        self.weight = weight
        self.sequence_noise = None
        self.dataset = dataset_name
        self.threshold = None
        self.noise_type = None
        self.classifier = weight_model 
    def norm_weight(self):
        self.weight = abs(self.weight)
        self.weight = self.weight/self.weight.sum(axis = 1, keepdims=1)
        return self.weight
    def add_noise_gaussian(self,noise_type=1,threshold = 50, k=1):
        if threshold == 0: 
            self.sequence_noise = self.sequence
        else:
            import random
            mu = 0
            pct_sigma = 0.02
            self.threshold = threshold
            self.noise_type = noise_type
            shape0,shape1 = len(self.sequence), len(self.sequence[0])
            range_ = (np.amax(self.sequence) - np.amin(self.sequence)) 
            sigma = range_*pct_sigma
            np.random.seed(2020)
            rand_matrix = np.random.randn(shape0,shape1)
            noise = sigma * rand_matrix + mu

            if threshold == 100: weight = np.full(shape = self.sequence.shape, fill_value = 1)
            else:
                if noise_type == 1: 
                    discrim = np.percentile(self.weight, 100-threshold, axis = 1).reshape(-1,1)
                    weight = (self.weight >= discrim) * 1

                elif noise_type == 2: 
                    discrim = np.percentile(self.weight, threshold, axis = 1).reshape(-1,1)
                    weight = (self.weight < discrim) * 1

            self.sequence_noise = self.sequence + np.multiply(weight*1.5*k,noise)

        return self.sequence_noise


    def add_noise_centroid(self, noise_type=1,threshold = 50):
        if threshold == 0: 
            self.sequence_noise = self.sequence
        else:
            self.threshold = threshold
            self.noise_type = noise_type
            
#             centroid_val = np.mean(self.sequence_noise)
            centroid_val = np.mean(self.train_x, axis = 0)

            if threshold == 100: discrim_area = np.full(shape = self.sequence.shape, fill_value = 1)
            else:
                if noise_type == 1: 
                    discrim = np.percentile(self.weight, 100-threshold, axis = 1).reshape(-1,1)
                    discrim_area = (self.weight >= discrim) * 1

                elif noise_type == 2: 
                    discrim = np.percentile(self.weight, threshold, axis = 1).reshape(-1,1)
                    discrim_area = (self.weight < discrim) * 1


            self.sequence_noise = discrim_area* centroid_val + self.sequence * (discrim_area == 0)

        return self.sequence_noise
    
    def visualize(self,idx):
        plt.figure(figsize = (10,6))
        plt.plot(self.sequence_noise[idx])
        plt.plot(self.sequence[idx])
        plt.legend(['Noisy', 'Original'], loc='lower right')
        plt.title('Signal with Type {} noise with threshold = {} at index {}'.format(self.noise_type, self.threshold, idx), fontdict = {'fontsize' : 12})

    def save(self):
        import utils.prep as prep
        
        txt = self.dataset + '_' + self.classifier + str(self.threshold) + '_type' + str(self.noise_type)
        
        prep.write_to_std(self.sequence_noise, self.label, dataset= txt, train = False)
    