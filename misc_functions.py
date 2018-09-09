#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Some global helper functions for all parts of the project.

Created on Mon Sep  3 15:19:56 2018
@author: nasir
"""

import numpy as np
import scipy.stats as stats


def get_data_location(is_quant=False):
    '''
    - get the location of deep net outputs. both natural and noise. This way, in
      we have changed the location of the data, we just have to change the
      location in only one place. Here!
    - data not supplied fo vgg. make sure your vgg network output is
      like : texture_natural_vggnet.mat etc
    - For the qualitative comparison, load the actual set (of 225 images) of data
      as done in the brain recording. But for the quantitative comparison, use the
      extended set (of 3375 images) of data. For the explanation for this data
      extension, please read the paper.
    '''
    if is_quant == True:
        file_prefix = '_extended'
    else:
        file_prefix = ''
    loc_natural = 'data/texture_natural%s_alexnet.mat' %file_prefix
    loc_noise = 'data/texture_noise%s_alexnet.mat' %file_prefix

    return loc_natural, loc_noise

def get_data_location_random(f_name):
    '''
    - data location for the random weight case. data size is different in this
      case since we do multiple (usually 10 for small data and 10 for extended)
      iterations and compute the modulation index on the fly. It saves us a huge
      amount of disk space.
    '''
    loc_rand_case = 'data/' + f_name
    return loc_rand_case

def cropCenterMap(lOutAll, nMid, nFilters, centerRespSize):
    '''
    - static method. as this is kind of an utility function. no need to create an
      object to call it
    - return the center (2x2 or 4x4 or whatever) map
    - crop center 2x2 maps from the all channels. e.g. convert from 225x128x27x27 -->
      225x512 (only center 2x2 of 128 channels)
    '''
    img_total = lOutAll.shape[0]
    offset = int(np.ceil(centerRespSize/2.0))    # how far go left to cover the size centerRespSize
    lCenter = lOutAll[:, :, nMid-offset:nMid+offset, nMid-offset:nMid+offset]   # 225x48x2x2 = 225x192
    lCenter = np.reshape(lCenter, [img_total, nFilters*centerRespSize*centerRespSize])
    return lCenter

def find_euclidean(from_cnn, target):
    """ Euclidean distance errors for both M_final and cross-val. From the V2."""
    vFreeman = target.reshape([-1])
    dist_euclid = np.linalg.norm(vFreeman - from_cnn)
    return dist_euclid

def find_correlation(from_cnn, target):
    """ find the correlation between CNN with bilogy  """
    corr_val = stats.pearsonr(target, from_cnn)
    return corr_val

def explained_var(from_cnn, target):
    """
    Explained variance.
    NOTE: This is equal to the corr_val^2
    """
    r, p = stats.pearsonr(from_cnn, target)
    r_squared = (r)**2
    return r_squared