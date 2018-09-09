#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- script to preprocess the data to cross validate and call the cross val functions
- needed for: greedy, optimal both usual and HMAX and ScatNet

Created on Sat Dec 30 21:51:28 2017
@author: nasir
"""
import numpy as np
import cvxpy as cvx
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from model_fit_common import ModelFit
import misc_functions


class CrossValidation(ModelFit):
    '''
    - inherit ModelFit class to use plot and Misc functions
    '''
    def init_cross_val(self, M, M_all, layerNum, **args):
        '''
        - process both usual and random cases
        - exclisively handle the 3375 image case. M is not averaged over samples. hard coded
          This is because, usual rand cases are saved after doing avg over samples except the 3375 images case
          because we need all image mod indices for cross validation
        '''
        is_random = args['is_random']
        nClasses = args['nClasses']
        t = args['t']
        v1_or_v2 = args['v1_or_v2']
        is_train = args['is_train']
        is_crossval = args['is_crossval']
        n_folds = args['n_folds']

        if is_random == True:
            n_iter = M_all.shape[0]
            predict_all = 0   # TODO: we dont need it in case of rand. just to avoid complain in rturn
            t_all = 0
            predict_mean_all = np.zeros((n_iter, nClasses), np.float64)
            euclid_error_dist_all = np.zeros(n_iter, np.float64)
            for i_t in range(n_iter):
                print('iteration %d ...' %(i_t+1))
                M_tmp_all = M_all[i_t, :, :]   # 3375x512 or 15x512
                # TODO: exclisively handle the 3375 image case. M is not averaged over samples. hard coded
                # This is because, usual rand cases are saved after doing avg over samples except the 3375 images case
                # because we need all image mod indices for cross validation
                if M_tmp_all.shape[0] == 3375:
                    nImgEachClass = int(M_tmp_all.shape[0] / nClasses)
                    M_tmp = np.reshape(M_tmp_all, [nClasses, nImgEachClass, -1])  # 15x225x512
                    M_tmp = np.squeeze(np.nanmean(M_tmp, 1))   # M_tmp is changed now. 15x512, avg over samples

                predict_mean_all[i_t, :], euclid_error_dist_all[i_t], _, _ = self.cross_validation(M_tmp, M_tmp_all, **args)

            predict_mean = np.mean(predict_mean_all, 0)
            euclid_error_dist = np.mean(euclid_error_dist_all)
            euclid_error_std = np.std(euclid_error_dist_all)
            r_squared = misc_functions.explained_var(np.mean(predict_mean_all, 0), t)  # explained variance, R-squared

            # now plot the fits. after avged over the iterations
            self.plotModelFitsCNNandBiology(predict_mean, t, np.mean(euclid_error_dist_all),
                                            v1_or_v2, is_train=is_train, is_crossval=is_crossval, zoom=0.12) # 0.23
            print('Cross-val error, L%d, (randomized, mean over %d iter), %d-fold: %0.4f' %(layerNum,
                    n_iter, n_folds, euclid_error_dist))
            print('Cross-val error for all %d iterations' %(n_iter))
            euclid_error_dist_all = np.around(euclid_error_dist_all, decimals=4)
            print(repr(euclid_error_dist_all))
            print('error standard dev: %0.4f' %euclid_error_std)
            print('Variance accounted for (R-squared): %0.4f' %r_squared)

        else:
            predict_mean, euclid_error_dist, predict_all, t_all = self.cross_validation(M, M_all, **args)
            self.plotModelFitsCNNandBiology(predict_mean, t, euclid_error_dist, v1_or_v2,
                                            is_train=is_train, is_crossval=is_crossval, zoom=0.12) # 0.23
            euclid_error_dist_all = np.copy(euclid_error_dist) # just to comply with return vals
            r_squared = misc_functions.explained_var(predict_mean, t)    # explained variance, R-squared
            print('Cross-val error, L%d, %d-fold: %0.4f' %(layerNum, n_folds, euclid_error_dist))
            print('Variance accounted for (R-squared): %0.4f' %r_squared)

        predict_mean = np.around(predict_mean, decimals=2)
        print(repr(predict_mean)) # print with comma separated
        return predict_mean, euclid_error_dist_all, predict_all, t_all

    def cross_validation(self, M, M_all, **args):
        '''
        input:
            M = 15x512
            M_all = 225x512 or 3375x512
        output:
            predict_mean = 15,
            euclid_error_dist = 1,
        in case 3375 images
        225 images in EACH class --> group 15 img --> 15 each class --> 225 in total --> predict 225 --
        --> avg over samples, 15 --> avg over iterations --> 15
        M_all = input data. all 225 images
        t = target. replicate t values untill 225. e.g. t1, t1. ...t1, t2, t2, ...
        idx_all should be [0, 1, 2, 3, ... ] for the case of t leave-one-out
        '''
        nClasses = args['nClasses']
        t = args['t']
        nNeurons = args['nNeurons']
        n_folds = args['n_folds']
        is_train = args['is_train']
        is_greedy = args['is_greedy']
        if is_greedy == False:
            lambdaVal = args['lambdaVal']

        nanIdx_all = np.argwhere(np.isnan(M))   # 511x2, will give the (row, col) of nan values
        nanIdx = np.unique(nanIdx_all[:, 1])    # unique col numbers i.e. neurons
        M_all = np.delete(M_all, [nanIdx], 1)   # 225x432, remove cols containing ALL images in a class. TODO: M_all is different for L2 rand
        M_all = np.nan_to_num(M_all)            # still has NaN. replace them with 0
        img_total = M_all.shape[0]
        nImgEachClass = img_total / nClasses

        # NOTE: following is in the case where we group 15 images, take their avg and continue the rests as usual.
        if nImgEachClass == 225:
            # Take avg of randomly 15 images in each class and fit the averages. it should smooth the predictions. Yes it does.
            n_img_each_group = 15       # take avg of group of 15 images. finally 225 images will be 15 images.
            n_after_nan = M_all.shape[1]
            M_all_new = np.reshape(M_all, [nClasses, nImgEachClass, n_after_nan]) #15x225x512
            M_all_new = np.mean(M_all_new.reshape(nClasses, n_img_each_group, nImgEachClass/n_img_each_group, n_after_nan), axis=2)
            M_all_new = M_all_new.reshape(nClasses*nImgEachClass/n_img_each_group, -1)
            img_total = M_all_new.shape[0]      # img_total now changed. or img_total = nClasses*nImgEachClass/n_img_each_group
            nImgEachClass = img_total/nClasses  # nImgEachImage is changed
            M_all = np.copy(M_all_new)          # continue with the M_all
        target = np.zeros(img_total, np.float32)    # 225,
        class_idx = np.zeros(img_total, np.int64)   # 225, values are the indices of target or class numbers, 0, 1, 2, .. 14
        tmp_t = t.reshape(-1)             # 15,  because, t might be 15x1 or 1x15. make sure its 15, size
        count = 0
        for iT in range(nClasses):
            val = tmp_t[iT]
            for iY in range(nImgEachClass):
                target[count] = val
                class_idx[count] = iT
                count = count + 1
        if is_greedy == True:
            predict_all, t_all, idx_all = self.CrossValidateGreedy_Kfold(
                    M_all, target.reshape(-1), class_idx, nNeurons=nNeurons,
                    n_folds=n_folds, is_train=is_train) # predict 225 or img_total, is_avg removed
        else:
            predict_all, t_all, idx_all = self.CrossValidateOptimal_Kfold(
                M_all, target.reshape(-1), nNeurons=nNeurons,
                n_folds=n_folds, lambd=lambdaVal, is_train=is_train) # predict 225 or the img_total
        # compute the mean prediction and plot with the target
        # keep track of the scrambles indices from the crossval function to return to the original t sequence
        back_to_orig = np.zeros((img_total), np.float64)
        back_to_orig[idx_all] = predict_all     # save according to the original order, shuffle was True
        predict_mean = np.mean(np.reshape(back_to_orig, [15, -1]), 1)    # [15, -1] classes
        euclid_error_dist = misc_functions.find_euclidean(np.array(predict_mean), t)
        return predict_mean, euclid_error_dist, predict_all, t_all


    def CrossValidateGreedy_Kfold(self, Mnew, t, class_idx, nNeurons=103, n_folds=9, is_train=False, is_avg=False):
        '''
        - Cross-validation. K-fold. Greedy
        - Mnew here is 225x432
        - is_train is for predicting on the trainig set itself. To see what is the best CNNs can do.
        - is_avg is the case, on k-fold case, where we take avg over all the classes of resp of images
          that is from 1200x432 --> 15x432 to both train and test and predict 15 values
        '''
        if is_train == True:
            n_folds = 1                         # predict the same examples the model trained on

        w = np.zeros(Mnew.shape[1])             # 432x1
        mse_pred = 0.0
        predict_all = np.array([], np.float32)  # 225 in total. 25 in each fold
        t_all = np.array([], np.float32)        # corresponding t values for what predictions are made in each fold/window
        idx_list_Mnew_all = np.zeros((n_folds, nNeurons), np.int32)    # 15x103 or 225x103, full populations
        idx_all = np.array([])                 # save the scrambled indices after Kfold
        # Just on the training sets. # only one fold. all 225 images. train and test prediction same
        if is_train == True:
            idx_all = range(Mnew.shape[0])      # just 0, 1, 2 ..., 225 in this case
            idx_list_Mnew = self.findIndexListGreedyFit(Mnew, nNeurons, t.reshape(1, -1))
            w[idx_list_Mnew] = 1.0/nNeurons
            pred_train = np.dot(Mnew, w)
            predict_all = np.append(predict_all, pred_train.flatten())
            t_all = np.append(t_all, t.flatten())
            mse_pred = mse_pred + self.mean_squared_error(t, pred_train)
            #print('mse predict. on Training images: %.4f' %(mse_pred))
        # prediction on the test, k-fold. n_folds can be until n-1
        elif (is_train == False and n_folds < Mnew.shape[0]):
            k_fold = KFold(len(Mnew), n_folds=n_folds, shuffle=True, random_state=None)    # use the same random seed
            for iK, (train, test) in enumerate(k_fold):
                #print('iK: ' + str(iK))
                idx_all = np.append(idx_all, test)  # save the indices after the input shuffled and folded
                # find the average of each folds across samples in each class. e.g. like 1200x432 --> 15x432
                if is_avg == True:
                    M_avg_train = self.avg_each_kfold(Mnew, train, class_idx) # convert from 1200x432 --> 15x432
                    M_avg_test = self.avg_each_kfold(Mnew, test, class_idx)
                    t_original = np.around(np.mean(np.reshape(t, [15, -1]), 1), decimals=4) # 15x1, just the t
                    idx_list_Mnew = self.findIndexListGreedyFit(M_avg_train, nNeurons, t_original.reshape(1, -1))
                    w[idx_list_Mnew] = 1.0/nNeurons
                    pred_test = np.dot(M_avg_test, w)
                    pred_train = np.dot(M_avg_train, w)
                    #print( "fold %d"%iK, np.sum(np.square(pred_train - pred_test)) )
                    #plt.scatter(pred_train, t_original); plt.show()
                else:
                    idx_list_Mnew = self.findIndexListGreedyFit(Mnew[train], nNeurons, t[train].reshape(1, -1)) # t has to be 1x25
                    w[idx_list_Mnew] = 1.0/nNeurons
                    pred_test = np.dot(Mnew[test], w)

                predict_all = np.append(predict_all, pred_test.flatten())
                t_all =  np.append(t_all, t[test].flatten())    # save a same arrangement of t for later
                idx_list_Mnew_all[iK, :] = idx_list_Mnew        # save the indices, check!
                w[:] = 0

            idx_all = idx_all.astype(int)   # indices have to be in int
        # leave-one-out. prediciton on the test, 225 fold, or 15 fold, same target sequence as train. is the one we are using!
        else:
            for iC in range(Mnew.shape[0]):     # 225 or 15 for the average predictions
                Mnew_test = Mnew[iC, :]                         # 1x432, ith row
                t_test = t[iC]
                Mnew_train = np.delete(Mnew, iC, axis=0)        # 224x432, remaining M, M_leave_one_out
                t_train = np.delete(t, iC, axis=0)              # 224, remove the corresponding value form the target
                # find w with 224 images instead of 225
                idx_list_Mnew = self.findIndexListGreedyFit(Mnew_train, nNeurons, t_train.reshape(1, -1)) # t has to be 1x14 here
                w[idx_list_Mnew]  = 1.0/nNeurons                # only those indices have 1.0/103, same values, rests are zeros

                pred_test = np.dot(Mnew_test, w)                # 1x1, prediction
                predict_all = np.append(predict_all, pred_test)
                t_all = np.append(t_all, t_test.flatten())
                idx_list_Mnew_all[iC, :] = idx_list_Mnew        # save the indices
                idx_all = np.append(idx_all, iC)                # indices from the input, should be same as orig target
                w[:] = 0
            idx_all = idx_all.astype(int)   # indices have to be in int
        return predict_all, t_all, idx_all


    def CrossValidateOptimal(self, Mnew, t, lambd=0.0):
        '''
        - find the cross-validation error. For optimal and regularized case.
        M = 15x432, modulation indices after removing NaN entries
        t = 15x1, target/biology modulation indces of 15 categories
        lambd = regularization parameter lambda
        say, t' = t after taking out the ith class and same for M

        in each iteraion/class
            w (or w_i) = 432x1, weights from training 14 classes (leaving one out) = argmin || t' - M' w ||^2
            trainErr = 1/14 || t' - M' w||^2
            valErr = (t[i] - M[i] w_i)^2

        train_sigma = very small
        valid_sigma =

        finally,
        train_error = 15x1, errors of each class. supposed to be very small
        valid_error = 15x1, validation errors
        '''
        w_all = np.zeros(np.shape(Mnew))                # 15x432, keep all the weights
        train_pred = np.zeros((Mnew.shape[0], 14))      # 15x14, 14 for each leaveOneOut case, i.e. in each folod
        t_leave_one_out = np.zeros((Mnew.shape[0], 14)) # 15x14, same
        valid_pred = np.zeros(Mnew.shape[0])
        train_error = np.zeros(Mnew.shape[0])           # 15x1
        valid_error = np.zeros(Mnew.shape[0])

        for iC in range(Mnew.shape[0]):
            M_i = Mnew[iC, :]                           # 1x432, ith row
            MleaveOneOut = np.delete(Mnew, iC, axis=0)  # 14x432, remaining M
            tleaveOneOut = np.delete(t, iC, axis=0)     # 14x1, remove the corresponding value form the target
            # find w with 14 classes instead of 15
            w = self.regularizedOptimization(MleaveOneOut, target=tleaveOneOut, lambd=lambd) # 432x1
            w_all[iC, :] = np.squeeze(w.value)          # keep this w if necessary

            pred_train_i = np.dot(MleaveOneOut, w.value)
            train_err_i = 1/14.0 * np.sum(np.asarray(tleaveOneOut - pred_train_i)**2)  # 1x1, Train error
            pred_i = np.dot(M_i, w.value)               # 1x1, prediction
            valid_err_i = (t[iC] - pred_i)**2           # 1x1, Test error, original - prediction
            train_pred[iC, :] = np.squeeze(pred_train_i)
            t_leave_one_out[iC, :] = np.squeeze(tleaveOneOut)
            valid_pred[iC] = pred_i
            train_error[iC] = train_err_i
            valid_error[iC] = valid_err_i

        return train_pred, valid_pred, t_leave_one_out, train_error, valid_error


    def regularizedOptimization(self, Mnew, target, lambd=0.0):
        '''
        Do the optimization with regularization.
        Mnew = M_new, after removing the NaN
        target = t, originally target is the t itself. But for cross-val, target is 14x1
        '''
        sizeW = np.size(Mnew, 1)         # based on the size of M. not fixed as we do not know how many NaN cols is gets deleted.
        w = cvx.Variable(sizeW)          # weights. to be computed using the optimization subject to the following constrains
        constraints = [0 <= w,  cvx.sum_entries(w) == 1] # constraints
        error = cvx.sum_squares(Mnew*w - target)
        reg = cvx.sum_squares(w)
        objective = cvx.Minimize(error + lambd * reg)
        prob = cvx.Problem(objective, constraints)  # solve for w
        prob.solve(solver='ECOS', verbose=False)
        return w

    def CrossValidateOptimal_Kfold(self, Mnew, t, nNeurons=103, n_folds=225, lambd=0.0, is_train=False):
        '''
        - Cross-validation. K-fold. Optimal
        - Mnew here is 225x432
        - is_train is for predicting on the trainig set itself. To see what is the best CNNs can do.
        - choose nNeurons only in case of the regularized.
        Mnew = 225x432, modulation indices after removing NaN entries
        t = 225, target/biology modulation indces of 15 categories
        lambd = regularization parameter lambda
        say, t' = t after taking out the ith class and same for M
        '''
        mse_train = 0.0
        mse_pred = 0.0
        predict_all = np.array([], np.float32)  # 225 in total. 25 in each folod
        t_all = np.array([], np.float32)        # corresponding t values for what predictions are made in each fold/window
        idx_all = np.array([])                  # save the scrambled indices after Kfold

        # Just on the training sets. # only one fold. all 225 images. train and test prediction same
        if is_train == True:
            idx_all = range(Mnew.shape[0])      # just 0, 1, 2 ..., 225 in this case
            w = self.regularizedOptimization(Mnew, target=t, lambd=lambd) # 432x1
            # Handle optimal and regularized case. do it here instead of calling findModindexMfinal
            if lambd == 0 or lambd == 0.0:      # optimal, weighted
                pred_train = np.dot(Mnew, w.value)          # same as pred_test in this case
            else:                               # Regularized. pick the highest 103 w indices. NOT weighted.
                topIndicesN = sorted(range(len(w.value)), key=lambda i: w[i].value, reverse=True)[:nNeurons]
                finalNeurons = np.array(topIndicesN[:])      # NaN already removed. so these are final
                pred_train = np.mean(Mnew[:, finalNeurons], 1)

            predict_all = np.append(predict_all, pred_train.flatten())
            t_all = np.append(t_all, t.flatten())
            mse_pred = mse_pred + mean_squared_error(t, pred_train)
            #print('mse predict. on Training images: %.4f' %(mse_pred))
        # prediction on the test, k-fold where k < 15, just a number
        elif (is_train == False and n_folds < Mnew.shape[0]):
            k_fold = KFold(len(Mnew), n_folds=n_folds, shuffle=True, random_state=1)    # use the same random seed
            for iK, (train, test) in enumerate(k_fold):
                idx_all = np.append(idx_all, test)   # save the indices after the input shuffled and folded
                w = self.regularizedOptimization(Mnew[train], target=t[train], lambd=lambd) # 432x1
                # Handle optimal and regularized case. do it here instead of calling findModindexMfinal
                if lambd == 0 or lambd == 0.0:       # optimal, weighted
                    pred_test = np.dot(Mnew[test], w.value)               # 1x1, prediction
                else:                                # Regularized. pick the highest 103 w indices. NOT weighted.
                    topIndicesN = sorted(range(len(w.value)), key=lambda i: w[i].value, reverse=True)[:nNeurons]
                    finalNeurons = np.array(topIndicesN[:])      # NaN already removed. so these are final
                    pred_test = np.mean(Mnew[test][:, finalNeurons], 1) # TODO: as in findModindexMfinal, check!

                predict_all = np.append(predict_all, pred_test.flatten())
                t_all = np.append(t_all, t[test].flatten())
                mse_pred = mse_pred + mean_squared_error(t[test], pred_test)
                pred_train = np.dot(Mnew[train], w.value)
                mse_train = mse_train + mean_squared_error(t[train], pred_train)
            idx_all = idx_all.astype(int)   # indices have to be in int
            mse_pred = mse_pred / n_folds
            mse_train = mse_train / n_folds
            #print('mse train and predict, optimal, K-fold: %.4f,  %.4f' %(mse_train, mse_pred))
        else: # 225 fold, leave only one image out
            for iC in range(225):
                idx_all = np.append(idx_all, iC)                # indices from the input, should be same as orig target
                Mnew_test = Mnew[iC, :]                            # 1x432, ith row
                t_test = t[iC]
                Mnew_train = np.delete(Mnew, iC, axis=0)        # 224x432, remaining M, M_leave_one_out
                t_train = np.delete(t, iC, axis=0)              # 224, remove the corresponding value form the target

                # find w with 224 images from 225
                w = self.regularizedOptimization(Mnew_train, target=t_train, lambd=lambd) # 432x1
                # Handle optimal and regularized case. do it here instead of calling findModindexMfinal
                if lambd == 0 or lambd == 0.0:                  # optimal, weighted
                    pred_test = np.dot(Mnew_test, w.value)      # 1x1, prediction
                else:                                # Regularized. pick the highest 103 w indices. NOT weighted.
                    topIndicesN = sorted(range(len(w.value)), key=lambda i: w[i].value, reverse=True)[:nNeurons]
                    finalNeurons = np.array(topIndicesN[:])     # NaN already removed. so these are final
                    pred_test = np.mean(Mnew_test[finalNeurons])# 1D matrix, Reg is NOT weighted as in findModindexMfinal
                predict_all = np.append(predict_all, pred_test.flatten())
                t_all = np.append(t_all, t_test.flatten())
                pred_train = np.dot(Mnew_train, w)
            idx_all = idx_all.astype(int)   # indices have to be in int
        return predict_all, t_all, idx_all


    def avg_each_kfold(self, Mnew, fold_idx, class_idx):
        '''
        - find the average of each folds across samples in each class. e.g. like 1200x432 --> 15x432
        - and the same for the test sets. keep track of the images coming from specific classes
        - number of images from each class arent gonna be the same, need to handle it when averagind
        - fold_idx is the list indices comes from each fold, train/test
        '''
        class_idx_fold = class_idx[fold_idx]            # find out the class indices for the samples
        M_curr_fold = Mnew[fold_idx]                    # train/test based on the call
        M_class_fold = np.zeros([15, Mnew.shape[1]])    # 15x432, after avg
        for iC in range(15):
            M_class_fold[iC, :] = np.mean(M_curr_fold[class_idx_fold == iC, ::])
        return M_class_fold

