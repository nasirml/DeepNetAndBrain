#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- plot variance ratio (1) as scatter plot (2) as proportion of units. later one is
  in the paper
- percent of units variable across families
- variance ratios
- ANOVA analysis

Compute the variance of L1 and L2 responses for textures.
    1. accross families (columns)
    2. accross samples (rows)
[ Following the paper: Ziemba, Freeman et. al., PNAS, 2016, page 3 ]

NOTE:
    you can use this code to analyse for any layer data from the deep network

Created on Thu Apr 19 13:10:18 2018
@author: nasir
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as stats
import random                  # to generate n unique random numbers in a range
import misc_functions
from init_params import InitParams

class VarianceRatio(InitParams):
    def __init__(self, k=1):
        '''
        Class variables. choose the values according to your needs. For example, if your
        data has more images in each class, change nImgEachClass or if you want to fit
        more than 103 neurons change n_neurons accordingly.
        '''
        super(VarianceRatio, self).__init__(k)
        self.nImgEachClass = 15

    def variance_ratio(self, is_quant=False):
        '''
        - variance ratio after one-way ANOVA
        - Deep network L1 units show more variability across samples in a same class
          whereas L2 units show more variability across families than samples. This
          trend of L2 versus L1 units is qualitatively similar to the perperties
          of V2 neurons, that V2 neurons are more sensitive to the different texture
          categorits/classes.
        '''
        params_dict = self.init_params()
        centerRespSize = params_dict['centerRespSize']
        n_neurons = params_dict['nNeurons']
        nClasses = params_dict['nClasses']
        # print what we are seeing in the current output
        print('Qualitative comparison (variance ratio) of brain V%d data with neurons in L%d layer of AlexNet'
              %(params_dict['v1_or_v2'], params_dict['layerNum']))

        loc_natural, _ = misc_functions.get_data_location(is_quant)
        allVarsDict = sio.loadmat(loc_natural)
        l1OutAll = allVarsDict['l1OutNatural']  # 225x48x54x54 --> 225x48x8x8 now!
        l2OutAll = allVarsDict['l2OutNatural']  # 225x128x27x27--> 225x128x8x8

        # L1
        nMid = int(l1OutAll.shape[-1] / 2)      # 54/2 = 27
        l1DataCenter = misc_functions.cropCenterMap(l1OutAll, nMid, l1OutAll.shape[1], centerRespSize) # 225x192
        randomNeurons = random.sample(range(0, l1DataCenter.shape[1]), n_neurons)   # 103 out of 192
        l1DataCenter = l1DataCenter[:, randomNeurons]  # 225x103
        l1DataCenter = np.reshape(l1DataCenter, [nClasses, self.nImgEachClass, -1]) # 15x15x103, samples and families
        [varSamplesL1, varFamilyL1] = self.one_way_ANOVA(l1DataCenter)

        # L2
        nMid = int(l2OutAll.shape[-1] / 2)      # 27/2 = 13
        l2DataCenter = misc_functions.cropCenterMap(l2OutAll, nMid, l2OutAll.shape[1], centerRespSize)  # 225x512
        #finalNeurons = sio.loadmat('finalNeuronsGreedyL2.mat')['finalNeurons'].flatten() # 103,
        randomNeurons = random.sample(range(0, l2DataCenter.shape[1]), n_neurons)   # 103 out of 512
        l2DataCenter = l2DataCenter[:, randomNeurons]     # 225x103
        l2DataCenter = np.reshape(l2DataCenter, [nClasses, self.nImgEachClass, -1]) # 15x15x103
        [varSamplesL2, varFamilyL2] = self.one_way_ANOVA(l2DataCenter)

        # percent of units variable across families
        l1VarFamily = int(np.ceil((float(np.sum(varFamilyL1 > varSamplesL1)) / varSamplesL1.shape[0]) * 100.0))
        l2VarFamily = int(np.ceil((float(np.sum(varFamilyL2 > varSamplesL2)) / varSamplesL2.shape[0]) * 100.0))
        print('Variable across families: ')
        print('L1: %d percent and L2: %d percent' %(l1VarFamily, l2VarFamily))

        # variance ratios
        var_ratio_l1 = self.find_ratio(varSamplesL1, varFamilyL1)
        var_ratio_l2 = self.find_ratio(varSamplesL2, varFamilyL2)
        print("Variance ratios: L1 = %0.2f, L2 = %0.2f" %(var_ratio_l1, var_ratio_l2))

        # t-tests, min=0.0 in L2 variance. giving problem in log
        p_val = self.t_test_in_log_domain(varSamplesL1, varFamilyL1, varSamplesL2, varFamilyL2)
        print("L1, L2 variance ratio p value (in log domain): %0.5f" %p_val)

        # NOTE: plots might look different in each run. because of the randomNeurons. But they are similar
        # (1) plot scatters
        self.plot_variance_ratio_scatter(varSamplesL1, varFamilyL1, layerNum=1)
        self.plot_variance_ratio_scatter(varSamplesL2, varFamilyL2, layerNum=2)

        # (2) plot variance ratio histograms in log scale. plots for the paper
        var_ratio_full_l1 = self.find_ratio(varSamplesL1, varFamilyL1, g_mean=False)
        var_ratio_full_l2 = self.find_ratio(varSamplesL2, varFamilyL2, g_mean=False)
        self.plot_variance_ratio_proportion(var_ratio_full_l1, layerNum=1, n_bins=12)
        self.plot_variance_ratio_proportion(var_ratio_full_l2, layerNum=2, n_bins=12)


    def one_way_ANOVA(self, l_data_center):
        '''
        i = sample index
        j = family index
        - we have the responses
           r1 = 15x225x48  is the l1DataCenter
           r2 = 15x225x128 is the l2DataCenter
        - return within sample and family variance
        '''
        mu_j = np.mean(l_data_center, axis=1, keepdims=True)
        gamma_ij = l_data_center - mu_j
        var_total = np.var(gamma_ij, axis=1)
        varSamplesL = np.mean(var_total, axis=0)
        varFamilyL = np.var(mu_j, axis=0)[0]
        # variance percentage for the plots
        norm_term = varSamplesL + varFamilyL
        varSamplesL = varSamplesL/norm_term * 100;
        varFamilyL = varFamilyL/norm_term * 100;
        varSamplesL = varSamplesL[np.logical_not(np.isnan(varSamplesL))] # remove nan
        varFamilyL = varFamilyL[np.logical_not(np.isnan(varFamilyL))]
        return varSamplesL, varFamilyL

    def find_ratio(self, var_samples, var_family, g_mean=True):
        ''' geometric mean variance in the original paper.'''
        var_ratio = var_family/var_samples
        if g_mean == True:
            return stats.mstats.gmean(var_ratio[np.logical_not(np.isnan(var_ratio))])
        else:
            return var_ratio    # send the element-wise ratio

    def t_test_in_log_domain(self, varSamplesL1, varFamilyL1, varSamplesL2, varFamilyL2):
        l1_ratio = np.log(varFamilyL1) / np.log(varSamplesL1)
        l2_ratio = np.log(varFamilyL2) / np.log(varSamplesL2)
        t_val, p_val = stats.ttest_ind(l1_ratio, l2_ratio)
        return p_val

    def plot_variance_ratio_scatter(self, varSamplesL, varFamilyL, layerNum, fontSize=14, lineWidth=1.3):
        fontTitle = 13
        dotSize = 65
        fig, ax1 = plt.subplots()
        ax1.plot([0, 95], [0, 95], c='k', lw=lineWidth)        # diagonal line.
        if layerNum == 1:
            ax1.scatter(varSamplesL, varFamilyL, s=dotSize, c='r', edgecolor='white')
        else:
            ax1.scatter(varSamplesL, varFamilyL, s=dotSize, c='b', edgecolor='white')
        ax1.set_xlabel('Variance across samples (%)', fontsize=fontSize)
        ax1.set_ylabel('Variance across families (%)', fontsize=fontSize)
        ax1.set_xlim([0, 102]); ax1.set_ylim([0, 102])
        ax1.set_xticks([0, 50, 100]); ax1.set_yticks([0, 50, 100])
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(lineWidth); ax1.spines['left'].set_linewidth(lineWidth)
        ax1.yaxis.set_ticks_position('left'); ax1.xaxis.set_ticks_position('bottom')
        ax1.tick_params(axis='both', direction='out', labelsize=fontSize, width=lineWidth)
        ax1.text(45, 80, 'L%d'%layerNum, fontsize=fontTitle, fontweight='bold', fontstyle='italic')
        plt.tight_layout(); plt.show();

    def plot_variance_ratio_proportion(self, var_ratio_full, layerNum, n_bins=15, fontSize=14, lineWidth=1.3):
        ''' variance ratio as proportion of units. the one in the paper  '''
        fig, ax = plt.subplots()
        edgeWidth = 0.50
        x_min = 0.01
        x_max = 10.0
        geom_mean = stats.mstats.gmean(var_ratio_full)
        log_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)
        weight = np.ones_like(var_ratio_full)/float(len(var_ratio_full))
        if layerNum == 1:
            plt.hist(var_ratio_full, bins=log_bins, normed=0,
                     weights=weight, color='green', edgecolor='w', linewidth=edgeWidth)
            plt.scatter(geom_mean, 0.37, color='green', marker='v')
        else:
            plt.hist(var_ratio_full, bins=log_bins, normed=0,
                     weights=weight, color='blue', edgecolor='w', linewidth=edgeWidth)
            plt.scatter(geom_mean, 0.37, color='blue', marker='v')
        ax.set_xlabel('Variance ratio', fontsize=fontSize)
        ax.set_ylabel('Proportion of units', fontsize=fontSize)
        ax.tick_params(axis='both', direction='out', labelsize=fontSize, width=lineWidth)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.axhline(linewidth=lineWidth, color='black'); ax.axvline(-0.26, linewidth=lineWidth, color='black')
        ax.axvline(1, linewidth=lineWidth, color='black')   # vertical line through the (0, 0)
        plt.gca().set_xscale('log')                         # x-axis is in log scale
        ax.set_xticks([0.1, 1.0, 1, 10]); ax.set_yticks([0.0, 0.2, 0.4])
        ax.text(0.02, 0.3, 'L%d'%layerNum, fontsize=fontSize, fontweight='bold', fontstyle='italic')
        plt.tight_layout(); plt.show()

