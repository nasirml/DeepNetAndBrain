#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- Functions to compute modulation index, one of the important metric in our project.
- Modulation index is important in both qualitative and quantitatice comparison.
  we use mod indices to fit deep network model layers to the brain data.

Created on Mon Sep  3 15:00:15 2018
@author: nasir
"""
import scipy.io as sio
import numpy as np
import misc_functions

def deep_net_mod_index(layerNum, fName, isVGG=False, centerRespSize=2, nClasses=15, is_quant=False):
    """
    - Compute modulation index M and M_all. In all cases including random
    - load data from ALL layers and then use the one needed for specific layers
    - MAT files contain the deep net outputs from the texture data. from all the layers.
    - random weight cases are handled separately since M is already computed. for 100 iterations
    - read the deep net output fro the extended version of data for quantitative
      comparison or cross validation.
    """
    if isVGG == False and layerNum < 10:    # AlexNet
        loc_natural, loc_noise = misc_functions.get_data_location(is_quant)
        allVarsDictNatural = sio.loadmat(loc_natural)
        allVarsDictNoise = sio.loadmat(loc_noise)

    if isVGG == True and layerNum < 10:     # VGG net
        # data not supplied. make sure your vgg network output is like : texture_natural_vggnet.mat etc
        loc_natural, loc_noise = misc_functions.get_data_location(is_quant)
        allVarsDictNatural = sio.loadmat(loc_natural)
        allVarsDictNoise = sio.loadmat(loc_noise)

    if layerNum < 10:
        lNaturalCenter, lNoiseCenter, nFilters = get_layer_data(allVarsDictNatural, allVarsDictNoise, layerNum, centerRespSize)
    else:
        loc_rand_case = misc_functions.get_data_location_random(fName)

    if layerNum == 11:
        M_dict = sio.loadmat(loc_rand_case)
        M_all = M_dict['M1_all']
        M = np.nanmean(M_all, 0)
    elif layerNum == 20 or layerNum == 21:
        M_dict = sio.loadmat(loc_rand_case)
        M_all = M_dict['M2_all']            # 100x15x512
        M = np.nanmean(M_all, 0)            # 15x512, avg over all the iterations or 3375x15
    elif layerNum == 321:
        M_dict = sio.loadmat(loc_rand_case)
        M_all = M_dict['M3_all']
        M = np.nanmean(M_all, 0)
    elif layerNum == 4321:
        M_dict = sio.loadmat(loc_rand_case)
        M_all = M_dict['M4_all']
        M = np.nanmean(M_all, 0)
    elif layerNum == 521:
        M_dict = sio.loadmat(loc_rand_case)
        M_all = M_dict['M2_all']
        M = np.nanmean(M_all, 0)
    else:
        nImgEachClass = int(lNaturalCenter.shape[0] / nClasses)     # find the num of image each class on the fly
        M, M_all = mod_index_compute(lNaturalCenter, lNoiseCenter, nClasses, nImgEachClass, nFilters, centerRespSize)
    return M, M_all
    
def mod_index_compute(lNaturalCenter, lNoiseCenter, nClasses, nImgEachClass, nFilters, centerRespSize):
    '''
    - function to actually compute the modulation index
    '''
    epsiLon = 1e-8                 # to save divide by zero error
    diffVal = lNaturalCenter - lNoiseCenter
    sumVal = lNaturalCenter + lNoiseCenter
    if np.sum(sumVal) == 0.0 or np.sum(sumVal) == 0:
        sumVal = sumVal + epsiLon
    modIndexTmp = diffVal / sumVal      # 225x512, has NaN entries
    M_all = np.copy(modIndexTmp)        # 225x512 is needed for the k-fold cross-validation
    modIndexTmp = np.reshape(modIndexTmp, [nClasses, nImgEachClass, nFilters*centerRespSize*centerRespSize])
    M = np.squeeze(np.nanmean(modIndexTmp, 1)) # 15x512, class x neurons, second 15 are samples and first 15 are families.

    return M, M_all    

def get_layer_data(allVarsDictNatural, allVarsDictNoise, layerNum, centerRespSize):
    l_natural = 'l%dOutNatural' %layerNum
    l_noise = 'l%dOutNoise' %layerNum
    l1OutAllNatural = allVarsDictNatural[l_natural]        # 225x48x54x54
    l1OutAllNoise = allVarsDictNoise[l_noise]
    nMid = int(l1OutAllNatural.shape[-1] / 2) # find nMid on the fly
    nFilters = int(l1OutAllNatural.shape[1])
    lNaturalCenter = misc_functions.cropCenterMap(l1OutAllNatural, nMid, nFilters, centerRespSize)
    lNoiseCenter = misc_functions.cropCenterMap(l1OutAllNoise, nMid, nFilters, centerRespSize)
    return lNaturalCenter, lNoiseCenter, nFilters


def hmax_mod_index(layerNum, nClasses, nImgEachClass):
    allVarsDictNatural = sio.loadmat('data/HmaxOutAllNatural3375_128.mat') # both c1 and c2
    allVarsDictNoise = sio.loadmat('data/HmaxOutAllNoise3375_128.mat')
    if layerNum == 1:
        hMaxOutNatural = allVarsDictNatural['c1OutNatural']         # 225x32x4x4
        hMaxOutNoise = allVarsDictNoise['c1OutNoise']
    else:
        hMaxOutNatural = allVarsDictNatural['c2OutNatural']         # 225x8x400
        hMaxOutNoise = allVarsDictNoise['c2OutNoise']
        # C2 output is |x - p_i|^2, distance. final output is exp(-beta*|x - p_i|^2)
        beta_natural = 1.0  # sharpness of the tunning
        beta_noise = 1.0
        hMaxOutNatural = np.exp(-beta_natural*hMaxOutNatural)
        hMaxOutNoise = np.exp(-beta_noise*hMaxOutNoise)
    modIdxM = (hMaxOutNatural - hMaxOutNoise) / (hMaxOutNatural + hMaxOutNoise)
    modIdxM = np.reshape(modIdxM, [nClasses*nImgEachClass, -1])     # c1:225x512,  c2:225x3200
    M_all = np.copy(modIdxM)                                        # 225x3200
    modIdxM = np.reshape(modIdxM, [nClasses, nImgEachClass, -1])    # 15x15x1536
    M = np.nanmean(modIdxM, 1)
    return M, M_all


def scatnet_mod_index(layerNum, centerRespSize, nClasses, nImgEachClass):
    # load the data. takes a while
    allVarsDictNatural = sio.loadmat('data/scatNetOut.mat') # both natural and noise
    scatNetOutNatural = allVarsDictNatural['scatNetOutNatural']         # 225x417x32x32
    scatNetOutNoise = allVarsDictNatural['scatNetOutNoise']             # 225x417x32x32
    nMid = int(scatNetOutNatural.shape[2]/2) # output size 225x417x32x32
    if layerNum == 2:
        nChannels = 384
        start = 33
        end = 417
    else:
        nChannels = 32
        start = 1
        end = 33
    m1Total = nChannels * centerRespSize * centerRespSize
    m1Natural = scatNetOutNatural[:, start:end, nMid-1:nMid+1, nMid-1:nMid+1]    # 225x32x2x2, index starts from 0
    m1Noise = scatNetOutNoise[:, start:end, nMid-1:nMid+1, nMid-1:nMid+1]
    modIdxM = (m1Natural - m1Noise) / (m1Natural + m1Noise)
    modIdxM = np.reshape(modIdxM, [225, m1Total]) # 225x128
    M_all = np.copy(modIdxM)        # 225x128
    modIdxM = np.reshape(modIdxM, [nClasses, nImgEachClass, m1Total])     # 15x15x128
    M = np.mean(modIdxM, 1)         # 15x128, only 15 classes vs all neurons response to that class
    return M, M_all

