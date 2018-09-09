#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- Greedy fit to neurophysiology data
-
NOTE:
    - dirPrefix, filePrefix not necessary to random 100 iteration case. only change the fName.
    -
Created on Mon Apr 17 15:39:06 2017
@author: nasir
"""

import numpy as np
import scipy.io as sio
from cross_val_common import CrossValidation
from init_params import InitParams
import misc_functions

class GreedyFit(InitParams, CrossValidation):
    # class hierarchy: ModelFit --> CrossVal --> GreedyFit
    # CrossVal inherits ModelFit. so GreedyFit automatically inherits ModelFit
    def __init__(self, k=1):
        '''
        - initialize the parameters and unpack them here. unpacking here makes them easier to use
          the variables in the later parts of this code. use self.nClasses instead of
          self.params_dict['nClasses'] everytime
        - So many params to allow you the flexibility to choose from any data directory,
          cross yes/no, various types of controls on the data, different layers of CNNs,
          fitting to V1 or V2 data, and much more!
        '''
        super(GreedyFit, self).__init__(k)
        params_dict = self.init_params()
        # print what we are seeing in the current output
        print('Quantitative comparison (Greedy) of brain V%d data with neurons in L%d layer of AlexNet'
              %(params_dict['v1_or_v2'], params_dict['layerNum']))

        self.v1_or_v2 = params_dict['v1_or_v2']
        self.layerNum = params_dict['layerNum']
        self.centerRespSize = params_dict['centerRespSize']
        self.is_crossval = params_dict['is_crossval']
        self.is_train = params_dict['is_train']
        self.n_folds = params_dict['n_folds']
        self.nNeurons = params_dict['nNeurons']
        self.is_avg = params_dict['is_avg']
        self.isVGG = params_dict['isVGG']
        self.nClasses = params_dict['nClasses']
        self.is_random = params_dict['is_random']
        self.t = params_dict['t']
        self.fName = params_dict['fName']

    def model_fit_greedy(self):
        '''
        - compute the global modulation index matrix M. 15x512, might contain NaN values
        - function calls. In random case, call 100 times for each iter/model and then take the avg of the errors
        - proces the extended dataset.
        '''
        M, M_all = self.findModindexM(
                self.layerNum, self.fName, isVGG=self.isVGG, centerRespSize=self.centerRespSize, nClasses=self.nClasses)

        if self.is_random == True:
            n_iter = M_all.shape[0]
            M_final_all = np.zeros((n_iter, self.nClasses), np.float64) # final predictions for all iterations
            euclid_error_dist_all = np.zeros(n_iter, np.float64)
            for i_t in range(n_iter):
                M_tmp = M_all[i_t, :, :]
                nImgEachClass = int(M_tmp.shape[0] / self.nClasses) # 225
                mod_idx_tmp = np.reshape(M_tmp, [self.nClasses, nImgEachClass, -1])  # 15x225x512
                M_tmp = np.squeeze(np.nanmean(mod_idx_tmp, 1))   # M_tmp is changed now. 15x512, avg over samples
                M_final_all[i_t, :], euclid_error_dist_all[i_t] = self.greedy_fit_modindex(M_tmp, M_all, is_random=self.is_random)
            euclid_error_std = np.std(euclid_error_dist_all)
            M_final = np.mean(M_final_all, 0)
            # Plot the fits with the biology. Based on the modulation index values
            self.plotModelFitsCNNandBiology(np.mean(M_final_all, 0), self.t, np.mean(euclid_error_dist_all), self.v1_or_v2, zoom=0.12) # 0.23

            print('Euclidean error, greedy fit (without cross-vall), L%d to V%d, center %dx%d: %0.4f'
                  %(self.layerNum, self.v1_or_v2, self.centerRespSize, self.centerRespSize, np.mean(euclid_error_dist_all)))
            print('error standard dev: %0.4f' %euclid_error_std)
            print('Greedy 103 units Avg Mod index (without cross-vall), L%d: %0.4f ' %(self.layerNum, np.mean(M_final)))
            print(repr(M_final))

        else:
            M_final, euclid_error_dist =  self.greedy_fit_modindex(M, M_all)

            # Plot the fits with the biology. Based on the modulation index values
            self.plotModelFitsCNNandBiology(M_final, self.t, euclid_error_dist, self.v1_or_v2, zoom=0.12) # 0.23 for the paper
            print('Greedy 103 units Avg Mod index (without cross-vall), L%d: %0.4f ' %(self.layerNum, np.mean(M_final))); print(repr(M_final))
            print('Euclidean error, greedy fit (without cross-vall), L%d to V%d, center %dx%d: %0.4f'
                  %(self.layerNum, self.v1_or_v2, self.centerRespSize, self.centerRespSize, euclid_error_dist))

        #--- cross validate k-fold, includeing leave-one-out. use M with 225x432 instead of 15x432
        # too many arguments. so pack them in a dictionary and send as one name.
        args = {'is_random': self.is_random, 'nClasses': self.nClasses, 't': self.t, 'is_train': self.is_train,
                'v1_or_v2': self.v1_or_v2, 'is_crossval': self.is_crossval, 'n_folds': self.n_folds, 'nNeurons': self.nNeurons,
                'is_greedy': True}
        if self.is_crossval == True:
            predict_mean, euclid_error_dist_all, predict_all, t_all = self.init_cross_val(M, M_all, self.layerNum, **args)
            # save all the values for rand cases. for t-test and R-squared
            #if self.layerNum == 20 or self.layerNum == 21 or self.layerNum ==321 or self.layerNum == 4321:
            #    sio.savemat('/home/nasir/dataRsrch/TexturesAlexNetOutAll%s/greedy_L%d_out.mat' %(self.dirPrefix, self.layerNum), {
            #    'predict_mean':predict_mean, 'euclid_error_dist_all':euclid_error_dist_all,
            #    'predict_all':predict_all, 't_all':t_all})
            # plot the original and predicted in the cv process
            #modelFitCommon.plotPredictAllScatter(t_all, predict_all, n_folds, is_train)
            #Training/test predictions. whatever comes from the above function. in one place with target
            #modelFitCommon.plotPredictAllLine(t_all, predict_all, n_folds, is_train)


    def greedy_fit_modindex(self, M, M_all, is_random=False):
        '''
        - do the greedy fitting and compute the euclidean distance
        - save the final neurons in case we need them

        input: 15x512 (for L2) mod index matrix. For L1, 15x192 etc
        output:
            - 15, mod index predictions (avg over neurons) from greedy fit using 103 'best' neurons
            - Euclidean error from this 15 predictions to the V2 units
        '''
        nanIdx_all = np.argwhere(np.isnan(M))   # 511x2, will give the (row, col) of nan values
        nanIdx = np.unique(nanIdx_all[:, 1])    # unique col numbers i.e. neurons
        M_new = np.delete(M, [nanIdx], 1)       # 15x432, after removing cols(neurons) with nan values
        #print('\nL%d and Mnew size: %d x %d'%(layerNum, M_new.shape[0], M_new.shape[1]) )
        #print('Avg mod index (from Mnew, not on 103 units): %.4f' %(np.mean(M_new)))
        # M_new indices are not matched with the 128x2x2. Following matrix is.
        # 5-6 : 5th index is basically the 6 th neuron in 128x2x2 plane/vector
        resultIndices = np.zeros(np.size(M_new, 1), np.int32) # index here is the index in M_new and value means index in M
        idx = 0; k = 0        # index in M and nanIdx
        l = 0                 # index in resultIndices
        for idx in range(np.size(M, 1)):     # travarse cols/neurons of M. To fill up the best neurons
            if k < np.size(nanIdx):          # done processing nanIdx or nanIdx is NULL
                if idx != nanIdx[k]:
                    resultIndices[l] = idx
                    l = l + 1
                else:
                    k = k + 1
            else:
                resultIndices[l] = idx
                l = l + 1
        if self.nNeurons > M_new.shape[1]:       # consider all neurons in case M_new does not thave that many
            nNeuronsAll = M_new.shape[1]
            idxListMnew = self.findIndexListGreedyFit(M_new, nNeuronsAll, self.t.reshape(1, -1))     # All neurons
        else:
            idxListMnew = self.findIndexListGreedyFit(M_new, self.nNeurons, self.t.reshape(1, -1))     # 103 unweighted best fit neurons
        # Now match the values from idxListMnew to M.
        # e.g. if idxListMnew[0] = 394, value in int(resultIndices[394]) is the original neuron number. Say its 463. Now look at the
        # 463th neuron in 128x2x2 (after flattening it).
        finalNeurons = resultIndices[idxListMnew]              # e.g. 394 is in idxListMnew[0]. So, finalNeurons[0] is 463
        #sio.savemat('finalNeuronsGreedyL%d.mat' %self.layerNum, {'finalNeurons': finalNeurons})

        avgForEachClassLTotal = M[:, finalNeurons]              # 15x103, pick up those 103 from M
        avgForEachClassL = np.nanmean(avgForEachClassLTotal, 1)     # 15x1, combine for all neurons. populatioin response
        avgForEachClassL = np.around(avgForEachClassL, decimals=2)  # convert to 2 decimal places. all values in the list
        M_final = np.copy(avgForEachClassL)     # just to make it similar as optimal code. There we use M_final
        euclid_error_dist = misc_functions.find_euclidean(M_final, self.t)

        return M_final, euclid_error_dist


