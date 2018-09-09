# -*- coding: utf-8 -*-
"""
- Find out the set of (103 L2) neurons that best describe the biology
- Solve the optimization problem
- Optimal weighted approach with constraint that weights add to 1
- Binarized version of optimal weighted approach, adding regularization term and thresholding.
-

- Algo
    M = 15x512. class x neurons. original data. mod index averaged over samples. still has some NaN
    M_new = 15x432 or something like it. after removing the NaN neurons from M
    t = 15x1, target. comes from biology
    w = 15x460 (based on the size of M_new)
    Now,
        Minimize ( || t - M_new * w ||^2 + lambda*|| w ||^2 )
        subject to sum(w) = 1 and w_i >= 0
    - find out the 103 highest values in w and corresponding neurons will be picked
- it is a Non-convex problem. Because of the negative lambda term. So firts try only the convex part
-

[1] Ping Li, S. S. Rangapuram and M. Slawski, Methods for sparse and low-rank recovery under simplex constraints.

Created on Fri Mar 17 16:24:24 2017
@author: nasir
"""

import numpy as np
import scipy.io as sio
from cross_val_common import CrossValidation
from init_params import InitParams
import misc_functions


class OptimalFit(InitParams, CrossValidation):
    def __init__(self, k=1, lambda_val=0.0):
        '''
        - lambda_val: 0=optimal weighted fit), 0.8=regularization (or any other value)
        - isNeuro: True = fit to neurophysiology data from Corey, 225x1 instead of 15x1 from the paper
        '''
        super(OptimalFit, self).__init__(k)
        params_dict = self.init_params()
        # print what we are seeing in the current output
        print('Quantitative comparison (optimal/regularized) of brain V%d data with neurons in L%d layer of AlexNet'
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

        self.lambda_val = lambda_val        # regularization parameter
        self.see_weights = False

    def model_fit_optimal(self):
        '''
        - compute the global modulation index matrix M. 15x512, might contain NaN values
        '''
        M, M_all = self.findModindexM(self.layerNum, self.fName, isVGG=self.isVGG,
                                      centerRespSize=self.centerRespSize, nClasses=self.nClasses)
        if self.is_random == True:
            n_iter = M_all.shape[0]
            M_final_all = np.zeros((n_iter, 15), np.float64) # final predictions for all iterations
            euclid_error_dist_all = np.zeros(n_iter, np.float64)
            for i_t in range(n_iter):
                M_tmp = M_all[i_t, :, :]
                nImgEachClass = int(M_tmp.shape[0] / self.nClasses)
                mod_idx_tmp = np.reshape(M_tmp, [self.nClasses, nImgEachClass, -1])  # 15x225x512
                M_tmp = np.squeeze(np.nanmean(mod_idx_tmp, 1))   # M_tmp is changed now. 15x512, avg over samples
                M_final_all[i_t, :], euclid_error_dist_all[i_t] = self.optimal_fit_modindex(M_tmp, M_all, is_random=self.is_random)
            euclid_error_std = np.std(euclid_error_dist_all)
            # Plot the fits with the biology. Based on the modulation index values
            self.plotModelFitsCNNandBiology(np.mean(M_final_all, 0), self.t, np.mean(euclid_error_dist_all), self.v1_or_v2, zoom=0.12)

            if self.lambda_val == 0.0:
                print('Euclidean error, optimal fit, L%d to V%d, center %dx%d: %0.4f'
                  %(self.layerNum, self.v1_or_v2, self.centerRespSize, self.centerRespSize, np.mean(euclid_error_dist_all)))
            else:
                print('Euclidean error, regularized fit, L%d to V%d, center %dx%d: %0.4f'
                  %(self.layerNum, self.v1_or_v2, self.centerRespSize, self.centerRespSize, np.mean(euclid_error_dist_all)))
            print('error standard dev: %0.4f' %euclid_error_std)

        else:
            M_final, euclid_error_dist =  self.optimal_fit_modindex(M, M_all)
            self.plotModelFitsCNNandBiology(M_final, self.t, euclid_error_dist, self.v1_or_v2, zoom=0.12)

            if self.lambda_val == 0.0:
                print('Euclidean error, optimal fit (without cross-vall), L%d to V%d, center %dx%d: %0.4f'
                  %(self.layerNum, self.v1_or_v2, self.centerRespSize, self.centerRespSize, euclid_error_dist))
            else:
                print('Euclidean error, regularized fit (without cross-vall), L%d to V%d, center %dx%d: %0.4f'
                  %(self.layerNum, self.v1_or_v2, self.centerRespSize, self.centerRespSize, euclid_error_dist))
            print('103 units Avg Mod index (without cross-vall), L%d: %0.4f ' %(self.layerNum, np.mean(M_final)))
            print(repr(M_final))

        # cross validate. k-fold and leave-one-out. use M with 225x432 instead of 15x432
        # too many arguments. so pack them in a dictionary and send as one name.
        args = {'is_random': self.is_random, 'nClasses': self.nClasses, 't': self.t, 'is_train': self.is_train,
                'v1_or_v2': self.v1_or_v2, 'is_crossval': self.is_crossval, 'n_folds': self.n_folds, 'nNeurons': self.nNeurons,
                'is_greedy': False, 'lambdaVal': self.lambda_val}

        if self.is_crossval == True:
            predict_mean, euclid_error_dist, predict_all, t_all = self.init_cross_val(M, M_all, self.layerNum, **args)
            # save all the values for rand cases. for t-test and R-squared
            #if self.layerNum == 20 or self.layerNum == 21 or self.layerNum ==321 or self.layerNum == 4321:
            #    sio.savemat('/home/nasir/.../regularized_L%d_out.mat' %(self.layerNum), {
            #    'predict_mean':predict_mean, 'euclid_error_dist_all':euclid_error_dist_all,
            #    'predict_all':predict_all, 't_all':t_all})
            # plot the original and predicted in the cv process
            #modelFitCommon.plotPredictAllScatter(t_all, predict_all, n_folds, is_train)
            #Training/test predictions. whatever comes from the above function. in one place with target
            #self.plotPredictAllLine(t_all, predict_all, self.n_folds, self.is_train)


    def optimal_fit_modindex(self, M, M_all, is_random=False):
        """
        - optimal fitting and compute the euclidean distance
        input: 15x512 (for L2) mod index matrix. For L1, 15x192 etc
        output:
            - 15, mod index predictions (avg over neurons) from optimal/reg fit using 103 'best' neurons
            - Euclidean error from this 15 predictions to the V2 units
        """
        nanIdx = np.argwhere(np.isnan(M))   # will give the (row, col) of nan values
        nanIdx = np.unique(nanIdx[:, 1])    # unique col numbers i.e. neurons
        M_new = np.delete(M, [nanIdx], 1)   # 15x432, refine M, deleting cols(neurons) having nan values.
        #print('\nL%d and Mnew size: %d x %d'%(layerNum, M_new.shape[0], M_new.shape[1]) )

        # M_new indices are not matched with the 128x2x2. Following matrix is.
        # 5-6 : 5th index is basically the 6 th neuron in 128x2x2 plan
        resultIndices = np.zeros(np.size(M_new, 1), np.int32)    # index here is the index in M_new and value means index in M
        idx = 0; k = 0          # index in M and nanIdx
        l = 0                   # index in resultIndices
        for idx in range(np.size(M, 1)):      # travarse cols/neurons of M. To fill up the best neurons
            if k < np.size(nanIdx):         # done processing nanIdx or nanIdx is NULL
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
        else:
            nNeuronsAll = self.nNeurons

        # optimization. w comes as cvx variable, lambd = 0.0 means no regularization
        w = self.regularizedOptimization(M_new, target=self.t.reshape(-1, 1), lambd=self.lambda_val)
        # maximum and minimum value and index of w
        #max_index, max_value = max(enumerate(w.value), key=operator.itemgetter(1))
        #min_index, min_value = min(enumerate(w.value), key=operator.itemgetter(1))

        # now pick the highest 103 w values and their indices
        topIndicesN = sorted(range(len(w.value)), key=lambda i: w[i].value, reverse=True)[:nNeuronsAll]
        topIndicesN_sorted = np.array(topIndicesN[:])   #  convert to array
        # Now match the values from topIndicesN to M. topIndicesN comes from the M_new
        # e.g. if topIndicesN[0] = 183. Now value in int(resultIndices[183]) is the original neuron number. Say its 202.
        # Now look at the 202th neuron in 128x2x2 (after flattening it).
        finalNeurons = resultIndices[topIndicesN_sorted]  # e.g. 183 is in topIndices_unsorted[41]. So, finalNeurons[41] is 202
        # Modulation indices. Both optimal and regularized mean
        M_final = self.findModindexMfinal(M, M_new, w, self.lambda_val, self.layerNum, finalNeurons, seeWeights=self.see_weights)

        # Euclidean distances.
        euclid_error_dist = misc_functions.find_euclidean(M_final, self.t)

        return M_final, euclid_error_dist



