#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- TSNE visualization of original 225 texture dataset
- Read the data saved from the network after supplying textures
  and visualize them

Created on Fri Jan 27 18:16:13 2017
@author: nasir
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    # for normalization before sending to tSNE
import scipy.io as sio
from bhtsne import tsne              # another implementation. sudo pip install bhtsne
import random
import misc_functions
from init_params import InitParams

class TsneVis(InitParams):
    def __init__(self, k=1):
        '''
        - Change: is_103, allVarsDict, centerRespSize
        - perplexity: balance attention between local and global aspects of the data.
          guess about the number of close neighbors each point has.
        - default for the perplexity value is 30. you can play around with other values
        - we have found that L2 neurons in AlexNet can separate out the different
          texture classes better than L1 neurons. This result is similar in our
          brain visual system, where V2 neurons show more texture selectivity
          than V1 neurons.
        '''
        super(TsneVis, self).__init__(k)
        self.perplexity = 30
        self.theta = 0.8

    def tsne_visualize(self, is_quant=False):
        '''
        '''
        params_dict = self.init_params()
        centerRespSize = params_dict['centerRespSize']
        # print what we are seeing in the current output
        print('Qualitative comparison (TSNE visualization) of brain V%d data with neurons in L%d layer of AlexNet'
              %(params_dict['v1_or_v2'], params_dict['layerNum']))

        is_103 = False          # True=choose 103. False=use full population
        loc_natural, _ = misc_functions.get_data_location(is_quant)

        allVarsDict = sio.loadmat(loc_natural)
        l1OutNatural = allVarsDict['l1OutNatural']          # now separate out each from the dictionary
        l2OutNatural = allVarsDict['l2OutNatural']
        labelsNatural = allVarsDict['labels']
        labelsNatural = labelsNatural.flatten()             # comes as 1x225, convert to (225, )

        nMid = int(l1OutNatural.shape[-1] / 2)  # 54/2 = 27
        l1CenterNatural = misc_functions.cropCenterMap(l1OutNatural, nMid, l1OutNatural.shape[1], centerRespSize)
        if is_103 == True:
            n_neurons = 103
            randomNeurons = random.sample(range(0, l1CenterNatural.shape[1]), n_neurons)   # 103 out of 192
            l1CenterNatural = l1CenterNatural[:, randomNeurons]  # 225x103

        nMid = int(l2OutNatural.shape[-1] / 2)
        l2CenterNatural = misc_functions.cropCenterMap(l2OutNatural, nMid, l2OutNatural.shape[1], centerRespSize)
        if is_103 == True:
            randomNeurons = random.sample(range(0, l2CenterNatural.shape[1]), n_neurons)
            l2CenterNatural = l2CenterNatural[:, randomNeurons]
        l1CenterNatural = preprocessing.scale(l1CenterNatural, axis=0)
        l2CenterNatural = preprocessing.scale(l2CenterNatural, axis=0)
        fig, ax = plt.subplots()
        self.plotTSNE(ax, l1CenterNatural, labelsNatural, self.perplexity, self.theta)
        fig, ax = plt.subplots()
        self.plotTSNE(ax, l2CenterNatural, labelsNatural, self.perplexity, self.theta)

    def plotTSNE(self, ax, tsneData, dataLabels, perplexity, theta=0.5):
        '''
        - plot and fencify
        '''
        lineWidth = 1.2
        tsneData = tsneData.astype('float64')
        tsneEmbedded = tsne(tsneData, perplexity=perplexity, theta=theta)
        plt.hsv()       # color model
        ax.scatter(tsneEmbedded[:, 0], tsneEmbedded[:, 1], s = 45,  c = dataLabels,
                   edgecolors='w', linewidth=0.30, alpha=0.75)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        #ax.yaxis.set_ticks_position('left'); ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        xmin, xmax = ax.get_xlim(); ax.axvline((xmax - np.abs(xmin))/2.0, color='black', linewidth=lineWidth)
        ymin, ymax = ax.get_ylim(); ax.axhline((ymax - np.abs(ymin))/2.0, color='black', linewidth=lineWidth)
        plt.tight_layout(); plt.show()
