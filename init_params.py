#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Initialize input parameters. Need this one for all the three (Greedy, Optimal,
and Reularized) fitting methods.

Created on Mon Aug 27 19:59:10 2018
@author: nasir
"""
import numpy as np
import scipy.io as sio
import csv
from itertools import islice

class InitParams(object):
    def __init__(self, k=1):
        self.params_dict = {}     # pack all the parameters into this dictionary
        self.kth_line = k

    def init_params(self):
        '''
        - return a pack of params to the GreedyFit/OptimalFit/RegFit after
        - handling the random cases. i.e. do the whole thing once only!
        - params has all the lines starting from k+1
        - just set your loc_natural and loc_noise in modelFitCommon.py and forget about the
          dirPrefix and filePrefix
        - handle random weights cases separately as we only need fName
        Descriptions:
         - v1_orv2: 1 = fit to V1 data, 2 = fit to V2 data
         - layerNum: 1=L1, 2=L2 etc and 11=L1rand, 20=L2rand, 21=L1L2rand,
           321, 4321, 521=shuffle L1L2
         - centerRespSize: center neighbor. default 2, or even number
         - is_train: usually False. True = predic on 225 same training images,
           False = prediction on K-fold
         - n_folds: usually 225, to leave one image out and train with 224, or 9 to leave 25 out
         - is_avg: whether we want to avg over samples in each fold. usually False
         - isVGG: True = VGG net, False = AlexNet
         -
        '''
        params = self.read_params()[0]# take only the first line from all lines starting from k
        #print(params)

        v1_or_v2 = int(params[0])
        layerNum = int(params[1])
        centerRespSize = int(params[2])
        is_crossval = int(params[3])
        is_train = int(params[4])
        n_folds = int(params[5])
        nNeurons = int(params[6])
        nClasses = int(params[7])

        # some other parameters you want to play around
        is_avg = False
        isVGG = False

        # random weights cases neede to handle separately.
        if layerNum > 10:
            is_random = True
        else:
            is_random = False

        # Target to fit.
        if v1_or_v2 == 1:
            t = np.array([-0.01, 0.03, -0.02, -0.04, -0.01, 0.015, -0.009, -0.05,
                          0.04, 0.04, -0.03, 0.09, 0.01, 0.025, -0.08]) # V1
        else:
            t = np.array([0.22, 0.175, 0.05, -0.02, 0.14, 0.115, 0.06, -0.01,
                          0.112, 0.20, 0.045, 0.23, 0.24, 0.20, 0.01]) # V2,
        '''
        - handle random cases. random L1 and others. All in one
        - 00 = output from L1; 20 or 21 = out from L2; 321=from L3; 4321=from L4
        - for 3375 or 225 images each class.
        '''
        if layerNum == 20:
            fName = 'random_l2_iter100_alexnet.mat' # for mod index, 225 img
            #fName = 'RandWeightsL2Iteration10_ContrastNorm.mat'  # for cross val, 3375 img
        elif layerNum == 11 or layerNum == 21 or layerNum == 321 or layerNum == 4321:
            fName = 'random_l1l2_iter100_alexnet'   # for mod index
            #fName = 'RandWeightsL1L2L3L4Iteration10_ContrastNorm.mat' # for cross-val
        elif layerNum == 521:
            fName = 'ShuffleWeightsL1L2Iteration100.mat'
        else:
            fName = ''
        self.params_dict = {'v1_or_v2':v1_or_v2, 'layerNum':layerNum, 'centerRespSize':centerRespSize,
                            'is_crossval':is_crossval, 'is_train':is_train,
                            'n_folds':n_folds, 'nNeurons':nNeurons,
                            'is_avg':is_avg, 'isVGG':isVGG, 'nClasses':nClasses,
                            'is_random':is_random, 't':t, 'fName':fName}
        return self.params_dict


    def read_params(self):
        '''
        - read input params for the project from a txt file
        - read just the k-th line, to see specific case outputs
        - we can also read all the lines starting from k
        '''
        params = []
        with open('params.txt', 'r') as fp:
             for line in islice(csv.reader(fp, delimiter=','), self.kth_line, None):
                params.append(line)
        return params
