#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Gateway to the:
    - Network output (step 1)
    - Qualitative comparison of deep net and brain (step 2)
    - Quantitative comparison of deep net and brain (step 3)

Created on Wed Sep  5 20:09:25 2018
@author: nasir
"""

from modulation_index import ModIndex
from tsne_visualization import TsneVis
from variance_ratio import VarianceRatio

from model_fit_greedy import GreedyFit
from model_fit_optimal import OptimalFit

class Qualitative(object):
    def __init__(self, kth_line):
        '''
        - Modulation index
        - TSNE visualization
        - Variance ratio
        '''
        self.ModIdx = ModIndex(k=kth_line)  # k is for the init_params class
        self.TsneVis = TsneVis(k=kth_line)
        self.VarRatio = VarianceRatio(k=kth_line)

class Quantitative(object):
    def __init__(self, kth_line, reg_param=0.0):
        '''
        - Greedy fit
        - Optimal fit, reg_param = 0
        - Regularized fit, reg_param > 0
        '''
        self.GreedyApproach = GreedyFit(k=kth_line)
        self.OptimalApproach = OptimalFit(k=kth_line, lambda_val=reg_param)
