#!/bin/python 

import numpy as np
import os
from lightgbm import LGBMClassifier
from sklearn.svm.classes import SVC
from sklearn.metrics.pairwise import chi2_kernel
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.utils import class_weight
from collections import Counter
import cPickle
import sys

# Apply the LGBM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    # Load the svm model
    lgbm = cPickle.load(open(model_file, 'r'))

    #file list for full or val
    #file_list = '/home/ubuntu/11775-hws/all_test.video'
    #file_list = 'list/test.video'
    file_list = '/home/ubuntu/11775-hws/all_test_fake.lst'
    #file_list = '/home/ubuntu/11775-hws/all_val.lst'

    # Create Test Video Label
    video_ids = []
    labels = []
    file_read_label = open(file_list, 'r')
    for line in file_read_label.readlines():
        video_id, label = line.strip().split(' ')
        video_ids.append(video_id)
        labels.append(label)
    file_read_label.close()

    # Read the input features
    if feat_dir == 'all/':
        X_df_k = []
        X_df_a = []
        for video_id in video_ids:
            file_path_k = 'kmeans/' + video_id + '.feature'
            if os.path.exists(file_path_k) is False:
                X_df_k.append(np.array([0]*180)) # The number of kmeans features
            else:
                X = np.loadtxt(file_path_k)
                X_df_k.append(np.array(X))

            file_path_a = 'asrfeat/' + video_id + '.feature'
            if os.path.exists(file_path_a) is False:
                X_df_a.append(np.array([0]*8006)) # The number of asrs features
            else:
                X = np.loadtxt(file_path_a)
                X_df_a.append(np.array(X))

        X_df_k = np.array(X_df_k)
        X_df_a = np.array(X_df_a)
        X_df = np.concatenate((X_df_k, X_df_a), axis = 1)
        X_df = np.array(X_df)
    else:
        X_df = []
        for video_id in video_ids:
            file_path = feat_dir + video_id + '.feature'
            if os.path.exists(file_path) is False:
                X_df.append(np.array([0]*feat_dim))
            else:
                X = np.loadtxt(file_path)
                X_df.append(np.array(X))

        X_df = np.array(X_df)
    
    #print(X_df.shape)
    
    #y_pred_proba = lgbm.decision_function(X_df)
    y_pred_proba = lgbm.predict_proba(X_df)
    lgbm.classes_
    
    np.savetxt(output_file, y_pred_proba[:,1])

    