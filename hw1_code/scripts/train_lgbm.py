#!/bin/python 

import numpy as np
import os
from lightgbm import LGBMClassifier
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.utils import class_weight
from collections import Counter
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        print "tuning -- grid search for parameter"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    tuning = sys.argv[5]

    #file_list = '/home/ubuntu/11775-hws/all_trn.lst'
    file_list = 'list/train'
    
    file_read = open(file_list, "r")
    X_df = []
    y_df = []

    for line in file_read.readlines():
        video_id, label = line.split(' ')
        label = label.replace('\n','')

        if feat_dir == 'all/':
            # Set the feature type (both mfcc and asr) and directory
            file_path_k = 'kmeans/' + video_id + '.feature'
            file_path_a = 'asrfeat/' + video_id + '.feature'

            if os.path.exists(file_path_k) is False:
                continue
            else:
                X_k = np.loadtxt(file_path_k)
                Y_label = label

            if os.path.exists(file_path_a) is False:
                continue
            else:
                X_a = np.loadtxt(file_path_a)
                Y_label = label

            if Y_label == event_name:
                y = 1
    #        elif Y_label == 'NULL':
    #            y = -1
            else:
                y = 0

            X = np.hstack([X_k, X_a])
            X_df.append(X)
            y_df.append(y)


        else: 
            # Set the feature type (mfcc or asr) and directory
            file_path = feat_dir + video_id + '.feature'

            if os.path.exists(file_path) is False:
                continue
            else:
                X = np.loadtxt(file_path)
                Y_label = label

            if Y_label == event_name:
                y = 1
    #        elif Y_label == 'NULL':
    #            y = -1
            else:
                y = 0

            X_df.append(X)
            y_df.append(y)

    file_read.close()   
    
    # Highly imbalanced data for training
    X_df = np.array(X_df)
    y_df = np.array(y_df)
    #print(Counter(y_df))
   
    #print(X_df.shape)
    
    # Add SMOTE resampling for dealing with imbalanced issue
    smote = SMOTE()
    X_df_res, y_df_res = smote.fit_sample(X_df, y_df)
    #print(Counter(y_df_res))    
    
    # Train lgbm
    clf = LGBMClassifier(n_estimators=500, class_weight='balanced', metric = 'binary_error', objective = 'binary', random_state=0)
#    clf.fit(X_df, y_df)    
    clf.fit(X_df_res, y_df_res)    

    if tuning =='False':
        # Save the LGBM model
        with open(output_file, 'w') as f:
            cPickle.dump(clf, f)    

        print 'LGBM trained successfully for event %s!' % (event_name)        
    else:
        import warnings
        warnings.filterwarnings("ignore")
        # Grid search for parameter tuning
        KF = KFold(n_splits=3, shuffle=True, random_state=0).split(X=X_df_res, y=y_df_res)

        param_grid = {
            'num_leaves': [30, 60, 90],
            'min_data_in_leaf': [50, 100, 200],
            'lambda_l1': [0, 1],
            'lambda_l2': [0, 1]
            }

        clf = LGBMClassifier(n_estimators=500, class_weight='balanced', metric = 'binary_error', objective = 'binary', random_state=0)

        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=KF)
        lgbm_model = grid_search.fit(X=X_df_res, y=y_df_res)

        print(lgbm_model.best_params_, lgbm_model.best_score_)

        clf_best = lgbm_model.best_estimator_
        clf_best.fit(X_df_res, y_df_res)
        clf = clf_best

        # Save the LGBM model
        with open(output_file, 'w') as f:
            cPickle.dump(clf, f)    

        print 'LGBM trained successfully for event %s!' % (event_name)