#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- generate the data/maps from the deep network to input in step 2 and step3
-

- NOTE:
    - output of this file, i.e the output from AlexNet is already supplied in
      the data section. So you do not need this unless you have different set of
      data or you want to test a different deep network model.
    - to save space, we cropped the center 8x8 since we only use the center
      neighborhood for all our experiments/simulations.
    - we downsample the images and then contrast normalize to make sure our
      receptive fields match the RFs in the brain experiments
    - Texture classes are (15 classes):
        D13  D18  D23  D30  D327  D336  D38  D393  D402  D48  D52  D56  D60  D71  D99
    - now we are doing with stride =2 instead of 4. So, load deploy_new.prototxt instead.
    - we use stride=2 in the first layer instead of 4 in the original AlexNet.
      that is why we use a different prototxt. You can change the network the
      way you want and use your own prototxt file
    - case: 1 = 256x256 center, 2 = padding, 3 = Buggy, 4 = contrast normalized
      5 = vals in [0, 255]
    -

Created on Jan 2017
@author: nasir
"""

import sys
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio  # savemat
import os               # listdir
#import caffe           # uncomment this line if you have Caffe installed
from skimage.measure import block_reduce   # for downsampling input image to 256x256

class DeepNet(object):
    def __init__(self, is_natural=True):
        self.is_natural = is_natural

    def save_deep_net_output(self):
        '''
        Change: is_natural, case, cnn_layers, l1Output(appr layerName), n_classes
                n_img_each_class
        '''
        case = 4
        n_classes = 15
        n_img_each_class = 15       # usually 15, 225 in case of the extended dataset,3375
        cnn_layers = ['norm1', 'norm2', 'conv3', 'conv4', 'pool5']
        alpha = 0.22                # 0.5 = binarize, 0.22 = contrast norm. desired standard deviation. 50 if imagee in [0, 255]
        beta = 0.5                  # desired luminance (gray). 127 if image in [0, 255]

        # setup the network and device
        caffe_root = '/home/nasir/caffe/'
        sys.path.insert(0, caffe_root + 'python')
        caffe.set_device(0)
        caffe.set_mode_gpu()
        model_def = 'deploy_new.prototxt'         # stride=2 in L1, instead of 4
        model_weights = '/home/nasir/dataRsrch/data1/bvlc_reference_caffenet.caffemodel'
        net = caffe.Net(model_def, model_weights, caffe.TEST)

        # load the mean of ImageNet (as distributed with Caffe) for subtraction
        mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')     # 3x256x256
        mu = mu.mean(1).mean(1) # avg over pixels to obtain the mean BGR pixels
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))   # move image channels to outermost dimensions
        transformer.set_mean('data', mu)               # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2, 1, 0))# RGB to BGR
        # set the size of the input (batch-size, image channels, image-size)
        net.blobs['data'].reshape(50, 3, 227, 227)

        # get the response map sizes on the fly
        resMapSizeL1 = net.blobs[cnn_layers[0]].data[0, :].shape[1]
        resMapSizeL2 = net.blobs[cnn_layers[1]].data[0, :].shape[1]
        resMapSizeL3 = net.blobs[cnn_layers[2]].data[0, :].shape[1]
        resMapSizeL4 = net.blobs[cnn_layers[3]].data[0, :].shape[1]
        resMapSizeL5 = net.blobs[cnn_layers[4]].data[0, :].shape[1]
        nFiltersL1 = 48
        nFiltersL2 = 128
        nFiltersL3 = 384
        nFiltersL4 = 384
        nFiltersL5 = 256

        if(self.is_natural == True):
            dbDir = '/home/nasir/dataRsrch/data1/Textures/naturalistic/'
            outFile = '/home/nasir/dataRsrch/data1/texture_natural_alexnet.mat'
        else:
            dbDir = '/home/nasir/dataRsrch/data1/Textures/noise/'
            outFile = '/home/nasir/dataRsrch/data1/texture_noise_alexnet.mat'
        classNames = os.listdir(dbDir)
        classNames = np.sort(classNames)  # lexicographically sort.

        if(self.is_natural == True):
            l1OutNatural = np.zeros((n_classes*n_img_each_class, nFiltersL1, resMapSizeL1, resMapSizeL1), dtype=np.float32)   # 225x48x54x54
            l2OutNatural = np.zeros((n_classes*n_img_each_class, nFiltersL2, resMapSizeL2, resMapSizeL2), dtype=np.float32)   # 225x128x27x27
            l3OutNatural = np.zeros((n_classes*n_img_each_class, nFiltersL3, resMapSizeL3, resMapSizeL3), dtype=np.float32)   # 225x384x27x27
            l4OutNatural = np.zeros((n_classes*n_img_each_class, nFiltersL4, resMapSizeL4, resMapSizeL4), dtype=np.float32)   # 225x384x27x27
            l5OutNatural = np.zeros((n_classes*n_img_each_class, nFiltersL5, resMapSizeL5, resMapSizeL5), dtype=np.float32)   # 225x256x13x13
            labelsNatural = np.zeros((1, n_classes*n_img_each_class))                       # labels
        else:
            l1OutNoise = np.zeros((n_classes*n_img_each_class, nFiltersL1, resMapSizeL1, resMapSizeL1), dtype=np.float32)
            l2OutNoise = np.zeros((n_classes*n_img_each_class, nFiltersL2, resMapSizeL2, resMapSizeL2), dtype=np.float32)
            l3OutNoise = np.zeros((n_classes*n_img_each_class, nFiltersL3, resMapSizeL3, resMapSizeL3), dtype=np.float32)
            l4OutNoise = np.zeros((n_classes*n_img_each_class, nFiltersL4, resMapSizeL4, resMapSizeL4), dtype=np.float32)
            l5OutNoise = np.zeros((n_classes*n_img_each_class, nFiltersL5, resMapSizeL5, resMapSizeL5), dtype=np.float32)
            labelsNoise = np.zeros((1, n_classes*n_img_each_class))

        count = 0           # img_total = n_classes*n_img_each_class
        for nC in range(n_classes):
            currClass = classNames[nC]
            print('processing class "%s" (%d of %d) ...' %(currClass, nC+1, n_classes))

            imgFileList = os.listdir(dbDir + currClass + '/')       # all files
            imgFileList = np.sort(imgFileList)                      # sor them as lexicographic order
            for nI in range(n_img_each_class):
                imgFile = imgFileList[nI]                           # tex-320x320-im13-smp1.png
                imgFileName = dbDir + currClass + '/' + imgFile     # ../tex-320x320-im13-smp1.png
                img = caffe.io.load_image(imgFileName)              # 320x320x3
                img = img[32:288, 32:288, :]                        # center 256x256x3
                if case == 1:                   # 1. Just the center 256x256
                    imgFinal = img
                    #plt.imshow(img)
                elif case == 2:                 # 2. Padding
                    imgDown = block_reduce(img, block_size=(2, 2, 1), func=np.mean)  # downsample to 128x128
                    imgPad = np.ones((256, 256, 3), dtype=np.float32)*0.5
                    imgPad[64:192, 64:192, :] = imgDown
                    imgFinal = imgPad
                    #plt.imshow(imgPad)
                elif case == 3:                 # 3. Contrast normalized
                    L = np.sum(img) / np.size(img)                           # luminance. np.size = rows * cols
                    #C = np.sqrt( np.sum(np.square(img - L)) / np.size(img)) # C, Actual
                    C = np.sqrt( np.square(img - L) )                        # Experiment with C
                    imgNormalized = alpha * (img - L) / C + beta             # check this matrix operation
                    imgFinal = imgNormalized
                    #plt.imshow(imgNormalized)
                elif case == 4:
                    '''
                    - the one we use for our paper. downsample to 128x128 --> 64x64 -->contrast norm
                    - pad surrounding with gray, hence 0.5, and make 256x256 to filt to network input requirement
                    '''
                    imgDown = block_reduce(img, block_size=(2, 2, 1), func=np.mean)  # downsample to 128x128
                    imgDown = block_reduce(imgDown, block_size=(2, 2, 1), func=np.mean)  # downsample to 64x64
                    L = np.mean(imgDown)                      # luminance. np.size = rows * cols
                    C = np.std(imgDown)    # contrast
                    imgNormalized = alpha * (imgDown - L) / C + beta
                    imgPad = np.ones((256, 256, 3), dtype=np.float32)*beta
                    imgPad[96:160, 96:160, :] = imgNormalized
                    imgFinal = imgPad
                elif case == 5:
                    imgFinal = img * 255
                else:
                    print('please select a CASE ...')

                transformed_image = transformer.preprocess('data', imgFinal)
                # copy the image data into memory allocated for the net
                net.blobs['data'].data[...] = transformed_image
                # perform classification
                output = net.forward()

                l1Output = net.blobs[cnn_layers[0]].data[0, 0:nFiltersL1]  # L1 output, first 48
                l2Output = net.blobs[cnn_layers[1]].data[0, 0:nFiltersL2]  # L2 output, first 128
                l3Output = net.blobs[cnn_layers[2]].data[0, :]             # L3 output, all of them
                l4Output = net.blobs[cnn_layers[3]].data[0, :]             # L4 output, all of them
                l5Output = net.blobs[cnn_layers[4]].data[0, :]             # L5 output, all of them

                # add them to the result matrix
                if(self.is_natural == True):
                    l1OutNatural[count, :, :, :] = l1Output
                    l2OutNatural[count, :, :, :] = l2Output
                    l3OutNatural[count, :, :, :] = l3Output
                    l4OutNatural[count, :, :, :] = l4Output
                    l5OutNatural[count, :, :, :] = l5Output
                    labelsNatural[:, count] = nC
                else:
                    l1OutNoise[count, :, :, :] = l1Output
                    l2OutNoise[count, :, :, :] = l2Output
                    l3OutNoise[count, :, :, :] = l3Output
                    l4OutNoise[count, :, :, :] = l4Output
                    l5OutNoise[count, :, :, :] = l5Output
                    labelsNoise[:, count] = nC
                count = count + 1

        # save the network outputs. only two layers to save space
        if(self.is_natural == True):
            sio.savemat(outFile, {'l1OutNatural':l1OutNatural,
                                  'l2OutNatural':l2OutNatural,
                                  #'l3OutNatural':l3OutNatural,
                                  #'l4OutNatural':l4OutNatural,
                                  #'l5OutNatural':l5OutNatural,
                                  'labels':labelsNatural})
        else:
            sio.savemat(outFile, {'l1OutNoise':l1OutNoise,
                                  'l2OutNoise':l2OutNoise,
                                  #'l3OutNoise':l3OutNoise,
                                  #'l4OutNoise':l4OutNoise,
                                  #'l5OutNoise':l5OutNoise,
                                  'labels':labelsNoise})

