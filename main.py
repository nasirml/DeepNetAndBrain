#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This code is the demonstration of our following paper:
Correspondence of Deep Neural Networks and the Brain for Visual Textures
Md Nasir Uddin Laskar, Luis G Sanchez Giraldo, and Odelia Schwartz
ArXiv Preprint. Link: https://arxiv.org/abs/1806.02888


NOTE:
    - Originally the program is written to work on the whole map from deep networks
      e.g. 225x48x54x54 for L1. But, to save space in the internet, we saved only the
      center 8x8 neighborhood instead of keeping 54x54. So in comment section, if you see
      54 (for L1) and 27 (for L2), just thing they are 8 !
    - Everything for AlexNet is done in this project. If you want to try any other
      deep network model, like VGG net, you can do it. Just get th network output
      by yourself, set the loc_natural and loc_noise as the location where you
      saved output maps and that's it!
    - Quantitative/cross-val data are resized to center 4x4 as the data size is
      too big to host in GitHub or download from a third party server.
    - random weight maps for cross validation is not included in the data directory
    -
Created on Mon Aug 27 15:11:02 2018
@author: nasir
"""
from gateway_to_all import Qualitative, Quantitative
from deep_net_output import DeepNet



def qualitative():
    '''
    - Qualitative correspondence of deep neural networks with the brain recording
      data as described in the paper.
    - You don NOT need to configure/install Caffe or TensorFlow to run this section.
      Because we have already supplied the deep net output data in the data/ directory.
    - Create instances of all the measures of qualitative correspondence so that
      we can invoke any metric just by using a member operator, instead of creating
      object each time.
    - k is to read specific line from file, k=2 means second line of the input
      which is the third line in the param.txt file. first line is the heading.
    '''
    k = 2
    QualitativeCorresp = Qualitative(kth_line=k)

    #QualitativeCorresp.ModIdx.modulation_index()
    QualitativeCorresp.TsneVis.tsne_visualize()
    #QualitativeCorresp.VarRatio.variance_ratio()

def quantitative():
    '''
    - Shows the quantitative correspondence of the deep network with the brain
      recording data, as described in the paper.
    - You don NOT need to configure/install Caffe or TensorFlow to run this section.
      Because we have already supplied the deep net output data in the data/ directory.
    - Read the necessary parameters from file and initialize them.
    - k is to read specific line from file, k=2 means second line of the input
      which is the third line in file. first line is the heading.
    - We use the maps from extended set of texture images for quantification,
      especially for the cross validation. Please go through the paper to know
      the reason for this. In short, a decent amount of data is necessary to do
      cross validation.
    - Note, the use of random case for cross-validation (layerNum>10 and is_crossval=1 in
      params.txt) take long time as we do the cross-validation in the extended dataset
      and we do 225 fold cross-validation 10 times and take the average of their results.
    - Layer L3 and L4 takes even more time as their data size is larger than L1 and L2
    '''
    k = 2
    reg_param = 0.0         # 0=optimal, >0 mean with regularization
    QuantitativeCorresp = Quantitative(kth_line=k, reg_param=reg_param)

    #QuantitativeCorresp.GreedyApproach.model_fit_greedy()
    QuantitativeCorresp.OptimalApproach.model_fit_optimal()


def deep_net_output():
    '''
    - Send the input images to the deep net, here we mostly use AlexNet, and find
      the outputs from each layer. We have included data as mat files so you do
      not need this step right away. You can use our data and run Steps 2 and 3
      and see the results described in the paper.
    - You need to have configured/installed the Caffe to run this section of code.
      if you do not have Caffe in your machine, you can safely ignore this part of
      the code as we have supplied the output of this section of the code in the
      data/ directory.
    - We have tested our method in many variants of deep network. For example, changing
      the strides in the layers, using random weights instead of pre-trained filters, etc.
    - But we provide data only for one specific case, you can manipulate the deep net
      as we describe int he paper (also as your own idea), send the texture images
      through the network, find the outputs and use our code to find the correspondence
      with the brain recording data.
    - Use your data --> send them in a deep network --> get the network outputs
      of different layers --> and go to step 2.
    '''
    is_natural = True
    deep_net_obj = DeepNet(is_natural=is_natural)
    deep_net_obj.save_deep_net_output()
    return 0


def main():
    '''
    step 1: Deep net output
    step 2: Qualitative comparison with the brain data
    step 3: Quantitative comparison with the brain data
    '''
    #deep_net_output()
    qualitative()
    #quantitative()


if __name__ == '__main__':
    main()