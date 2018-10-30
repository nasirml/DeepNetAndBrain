#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- compute modulation index.
- mod index of all the families/classes averaged over the number of samples/images
  in each class
- computed on the response map from the deep network after textures are used as
  the input. any layer output.
- Formula
    M = (response_natural - response_noise) / (response_natural + response_noise)

NOTE:
    - here, mod index is computed after the NaN values are deleted.
    - Final result slightly varies because of the randomNeurons. In a separate run,
      different rendomNeurons are selected.

Updated on Dec 11, 2017
@author: nasir
"""
import numpy as np
import random               # to generate n unique random numbers in a range
import modulation_index_common
from init_params import InitParams

class ModIndex(InitParams):
    def __init__(self, model_select=1, n_iteration=1000, k=1):
        '''
        - k is the parameter of the superclass
        - model_select: 1=CNN, 2=Hmax, 3=ScatNet
        - n_iteration: multiple random iteration and then take the average value
        '''
        super(ModIndex, self).__init__(k)
        self.model_select = model_select
        self.n_iteration = n_iteration


    def modulation_index(self):
        '''
        - get the specific parameters using the init_params
        - refine M, deleting cols(neurons) having nan values.
        - compute the global modulation index matrix M. 15x512, might contain NaN values
        - in the brain, V1 modulation index is lower around 0 and
          V2 mod index is higher, around 0.12. we have seen the smilar trend in
          deep network layers, where L2 mod index is higher than L1.
        '''
        params_dict = self.init_params()

        layerNum = params_dict['layerNum']
        centerRespSize = params_dict['centerRespSize']
        nNeurons = params_dict['nNeurons']
        nClasses = params_dict['nClasses']
        fName = params_dict['fName']
        is_random = params_dict['is_random']
        nImgEachClass = 15

        # print what we are seeing in the current output
        print('Qualitative comparison (mod index) of brain V%d data with neurons in L%d layer of AlexNet'
              %(params_dict['v1_or_v2'], params_dict['layerNum']))

        if self.model_select == 1:
            M, M_all = modulation_index_common.deep_net_mod_index(layerNum, fName, isVGG=False,
                                                                  centerRespSize=centerRespSize,
                                                                  nClasses=nClasses)
        elif self.model_select == 2:
            M, M_all = modulation_index_common.hmax_mod_index()
        else:
            M, M_all = modulation_index_common.scatnet_mod_index()

        avgForEachClassL = self.cnn_mod_index_refine(M, nClasses=nClasses, nImgEachClass=nImgEachClass,
                                        nNeurons=nNeurons, is_random=is_random)

        print('\nAvg Mod index, L%d: %0.2f ' %(layerNum, np.mean(avgForEachClassL)))
        print(repr(avgForEachClassL))       # convert to string and print with commas

        # check if the order matches with the bilogy
        if layerNum == 2:
            order = [i[0] for i in sorted(enumerate(avgForEachClassL), key=lambda x:x[1])][::-1]
            print('Order (L2, from large to small mod index): ')
            # Freeman reproduce order is: [12, 11, 0, 9, 13, 1, 4, 5, 8, 6, 2, 10, 14, 7, 3] compare
            print(order)

    def cnn_mod_index_refine(self, M, nClasses=15, nImgEachClass=15, nIteration=10000, nNeurons=103, is_random=False):
        '''
        - address the NaN values, do the averageing over the samples and
          families, handle decimal places
        '''
        nanIdx_all = np.argwhere(np.isnan(M))   # 511x2, will give the (row, col) of nan values
        nanIdx = np.unique(nanIdx_all[:, 1])    # unique col numbers i.e. neurons
        M_new = np.delete(M, [nanIdx], 1) # 15x432, refine M, deleting cols(neurons) having nan values. idices and from cols=1

        # 225x512 --> 15x512
        if is_random == True:
            if M_new.shape[0] == 225:
                mod_idx_tmp = np.reshape(M_new, [nClasses, nImgEachClass, -1])  # 15x15x512
                M_new = np.squeeze(np.nanmean(mod_idx_tmp, 1)) # 15x512

        neurons_total = M_new.shape[1]
        avgForEachClassLTotal = np.zeros((nIteration, nClasses))
        for iT in range(nIteration):
            randomNeurons = random.sample(range(0, neurons_total), nNeurons)   # 103x1
            avgForEachClassTmp = np.mean(M_new[:, randomNeurons], 1)
            avgForEachClassLTotal[iT, :] = avgForEachClassTmp

        avgForEachClassL = np.mean(avgForEachClassLTotal, 0)        # 15x1
        avgForEachClassL = np.around(avgForEachClassL, decimals=4)  # convert to 2 decimal places. all values in the list
        return avgForEachClassL


