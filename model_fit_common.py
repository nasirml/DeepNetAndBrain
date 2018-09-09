#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- All common functions for model fits: greedy, optimal, regularized
Functions to:
    - Find mod index matrix M, for any case, any layer
    - compute the fits
    - misc. e.g. compute Euclidean distance
-

Created on Thu Aug 24 17:28:04 2017
@author: nasir
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmat  # to compute the argmax of min pair-wise distance
from matplotlib.offsetbox import AnnotationBbox, OffsetImage # put texture image as marker
from PIL import Image
import os
import modulation_index_common


class ModelFit(object):
    def __init__(self):
        self.is_quant = True     # to read the extended version of data

    #% find the main modulation index matrix M
    def findModindexM(self, layerNum, fName, isVGG=False, centerRespSize=2, nClasses = 15):
        """
        - Fits are based on the modulation indices. So find the modulation index
          first and then start fiting and crossvalidating.
        - Finding the modulation index is same as we do it in the qualitative
          correspondence case.
        - Compute modulation index M and M_all. In all cases including random
        - set is_quant to True as we want to read the extended set of maps
        """
        M, M_all = modulation_index_common.deep_net_mod_index(layerNum, fName, isVGG=False,
                                                              centerRespSize=2, nClasses=15,
                                                              is_quant=True)
        return M, M_all

    def choose_some_images(self, M_all, target, how_many=50):
        '''
        - in case we need to choose some images from each class.
          Instead of all the images which we do usually
        '''
        idx_img = np.array([], np.int32)
        i_count = 0
        for iImg in range(15):
            i_idx = range(i_count, i_count+how_many)
            idx_img = np.append(idx_img, i_idx)
            i_count = i_count + 200
        M_all = M_all[idx_img, :]
        target = target[idx_img]
        return M_all, target

    # Fits: Greedy
    def findIndexListGreedyFit(self, Mnew, nNeurons, t):
        '''
        - Greedy fit
        - Find the indices based on the minimum distance
        - Compute minimum distances between one point and a set of points. computes for each row in X, the index of the row of Y
          which is closest (according to the specified distance). by default Euclidean distance. can be any other distances.
          Start with the WHOLE M_new and gradually take out the best fit neurons. i.e. take indices from remainingList and put into
          idxList. idxList contains the indices of final 103 neurons.
          idxListMnew is the index on M_new matrix, 432 size. we have to convert this to M matrix which is 512 size
        - remainingList: start with WHOLE M_new matrix and gradually take out best fit neuron
        '''
        Mnew = np.transpose(Mnew)             # 432x15, Returns row index, so transpose M_new. Now rows are the neurons.
        idxListMnew = np.array([], np.int32)  # will be 103x1, final neurons list on M_new. append one by one in this pool
        remainingList = np.array(range(Mnew.shape[0]))# 432x1,
        for i in range(nNeurons):
            M_tmp = Mnew[remainingList, :]
            if i != 0:
                for j in range(remainingList.shape[0]):
                    M_tmp[j, :] = (M_tmp[j, :] + np.sum(Mnew[idxListMnew, :], 0)) / (idxListMnew.shape[0] + 1)
            nan_count = np.sum(np.isnan(M_tmp))
            if nan_count > 0:
                print('NaN count: ' +str(nan_count))
            idx, val = skmat.pairwise_distances_argmin_min(t, M_tmp)    # pair-wise minimum distance and index
            minIdxActual = remainingList[idx][0]                        # Index from M_new instead of M_tmp
            idxListMnew = np.append(idxListMnew, minIdxActual)          # put the index on the pool.
            idxUsed = np.where(remainingList == minIdxActual)[0][0]     # comes with array so get just the value
            remainingList = np.delete(remainingList, idxUsed);          # take out the value from the list
        return idxListMnew

    # Fits: weighted optimal and regularized
    def findModindexMfinal(self, M, M_new, w, lambdaVal, layerNum, finalNeurons, seeWeights=False):
        '''
        - Optimal and Regularized fit
        - using the w values computed from regularization function, compute the M_final
        - optimal Weighted approach with sum(w) = 1. consider ALL neurons
        '''
        if lambdaVal == 0 or lambdaVal == 0.0:
            w_new = np.zeros(np.shape(w.value))         # 432x1
            w_new = np.copy(w.value)            # use ALL the 432 neurons instead of top 103
            M_final = np.dot(M_new, w_new)              # 15x1, is the avg modulation index now
            M_final = np.around(M_final, decimals=2)    # convert to 2 decimal points
            M_final = np.transpose(np.squeeze(M_final))
            # see the weights
            if seeWeights == True:
                fig, ax = plt.subplots()
                plt.plot(w.value)           # unweighted regularized weights
                plt.show()
        else:   # weighted approach with regularization and thresholding. ONLY top 103
            M_final = np.mean(M[:, finalNeurons], 1)
            M_final = np.around(M_final, decimals=2)
            # see the weights
            if seeWeights == True:
                fig, ax = plt.subplots()
                plt.plot(w.value)           # unweighted regularized weights
                plt.show()
        return M_final

    # Plots: Fits, cross-validates,
    def plotModelFitsCNNandBiology_Neuro(self, M_final, target, layerNum, v1_or_v2):
        vFreeman = target.reshape([-1])
        plt.scatter(vFreeman, M_final, s=65, color='r', edgecolor='w')
        plt.xlabel('Biology, all images, V%d' %v1_or_v2); plt.ylabel('Modulation in CNN, L%d' %layerNum);


    def plotModelFitsCNNandBiology(self, M_final, target, euclid_error, v1_or_v2, is_train=False, is_crossval=False, zoom = 0.12):
        '''
        - M_final is the avg mod index values from any layers
        - zoom = 0.12 usual and 0.07 while uisng latexify
        '''
        euclid_error_dist = euclid_error
        euclid_error_dist = np.around(euclid_error, decimals=2)
        vFreeman = target.reshape([-1])
        fig, ax = plt.subplots()
        plt.gray()
        font_size = 15         # 15 if latexify, 50 for the paper
        line_width = 1.0       # line thickness. 1.0 usual, 4.0 for the paper

        # Texture images to be plot as marker
        xMin = -0.08; yMin = -0.08
        if v1_or_v2 == 1:
            xMax = 0.1; yMax = 0.1
        else:
            xMax = 0.3; yMax = 0.3

        imgDir = '/home/nasir/dataRsrch/data1/samples/'
        imgFilesTextures = np.sort(os.listdir(imgDir))
        # find the order indices. need this part to maintain the order in V2
        order = [i[0] for i in sorted(enumerate(M_final), key=lambda x:x[1])][::-1]
        M_final = np.sort(M_final)[::-1]      # sort descending order.
        vFreeman = vFreeman[order]    # re-arrange them according to the sorted order

        idx = 0
        for x0, y0 in zip(vFreeman, M_final):
            #img = plt.imread(imgDir + imgFilesTextures[order[idx]])
            #img = mpimage.imread(imgDir + imgFilesTextures[order[idx]])
            img = Image.open(imgDir + imgFilesTextures[order[idx]]).convert('L') # maintains the mod index value order
            idx = idx + 1
            imgBox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imgBox, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.scatter(vFreeman, M_final)       # plot only L2 vs V2 as texture marker
        ax.set_xlim([xMin, xMax]); ax.set_ylim([yMin, yMax]);
        if v1_or_v2 == 1:
            ax.set_xticks([0.0, 0.05, 0.1]);        # V1
            ax.set_yticks([0.0, 0.05, 0.1]);
        else:
            ax.set_xticks([0.0, 0.1, 0.2, 0.3]);   # V2
            ax.set_yticks([0.0, 0.1, 0.2, 0.3]);
        #ax.xaxis.set_ticklabels([])                 # just ticks but NO labels
        #ax.yaxis.set_ticklabels([])                 # just ticks but NO labels
        ax.plot([xMin, xMax-0.03], [yMin, yMax-0.03], color = 'black', linewidth=line_width, linestyle='--')
        ax.set_xlabel('Modulation in Biology, V%d' %v1_or_v2, fontsize=font_size);

        if is_crossval == True:
            if is_train == True:
                ax.set_ylabel('Crossval train in CNN', fontsize=font_size)
            else:
                print(' ')
                #ax.set_ylabel('Crossval test in CNN', fontsize=font_size)
                #ax.set_ylabel('Modulation in CNN', fontsize=font_size)
        else:
            print(' ')              # to avoid null conditioning
            ax.set_ylabel('Modulation in CNN', fontsize=font_size);
            #ax.set_ylabel('Modulation in CNN, L2', fontsize=font_size);
            #ax.set_ylabel('Modulation in ScatNet', fontsize=font_size);
            #ax.set_ylabel('Modulation in HMAX', fontsize=font_size);
            #ax.text(0.0, 0.23, r'\textbf{L2 rand}', fontsize=font_size)  # LaTeX style text in the plot

        ax.tick_params(axis='both', direction='out', labelsize=font_size-2, length=line_width+2, width=line_width)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left');
        ax.spines['bottom'].set_linewidth(line_width); ax.spines['left'].set_linewidth(line_width) # axes linewidth
        #ax.invert_xaxis()   # x-axis reversed
        ax.text(0.0, 0.25, r'E = %0.2f' %euclid_error_dist, fontsize=font_size)
        #ax.text(0.15, 0.0, r'E = %0.2f' %euclid_error_dist, fontsize=font_size)
        plt.tight_layout(); plt.show();


    # Training/test predictions. whatever comes from the above function. in one place with target
    def plotPredictAllLine(self, t_all, predict_all, n_folds=9, is_train=False):
        font_size = 18
        line_width = 1.75

        x_max = np.size(predict_all)
        fig, ax = plt.subplots()
        if is_train == False:
            plt.plot(predict_all, color='green', label='predicted on test')
            #plt.ylabel('Predicted on the Test sets, %d-fold' %n_folds)
            plt.ylabel('Predicted on the test sets', fontsize=font_size)
        else:
            plt.plot(predict_all, label='predicted on train')
            plt.ylabel('Predicted on the training sets', fontsize=font_size);

        plt.plot(t_all, 'r', label='target biology')
        plt.xlabel('Texture stimulus', fontsize=font_size);
        if x_max == 225:        # all 225 images
            plt.xticks([0, 50, 100, 150, 200]);
        elif x_max >= 1000:
            plt.xticks([0, 500, 1000, x_max])
        else:                   # take average, only 15 classes
            plt.xticks([0, 5, 10, 15]);

        plt.yticks([-0.1, 0.0, 0.2, 0.4])
        ax.tick_params(axis='both', direction='out', labelsize=font_size-2, width=line_width)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(line_width); ax.spines['left'].set_linewidth(line_width)
        legnd = plt.legend(loc='upper left', fontsize=font_size)
        for legobj in legnd.legendHandles:
            legobj.set_linewidth(line_width)        # legend linewidth
        plt.tight_layout(); plt.show()


    # plot the original and predicted in the cv process, all 225 predictions vs target
    def plotPredictAllScatter(self, t_all, predict_all, n_folds=9, is_train=False):
        plt.subplots()
        plt.scatter(t_all, predict_all, s=65, edgecolor='white')
        plt.xlabel('Biology, augmented to %d' %t_all.shape[0]);
        if is_train == False:
             plt.ylabel('Predicted on the Test sets, %d-fold' %n_folds)
        else:
             plt.ylabel('Predicted on the same training images')
        #plt.xlim([0, 0.3]); plt.ylim([0, 0.3])
        plt.tight_layout(); plt.show()

