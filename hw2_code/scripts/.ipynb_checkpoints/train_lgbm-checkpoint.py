#!/bin/python 

import numpy as np
import os
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.utils import class_weight
from collections import Counter
import pickle
import sys

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass
warnings.warn = warn

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print ("Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0]))
        print ("event_name -- name of the event (P001, P002 or P003)")
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features")
        print ("data -- train only or train_val (train+val)")
        print ("output_file -- path to save the svm model")
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    data = sys.argv[4]
    output_file = sys.argv[5]
    
    if data == 'train':
        file_list = 'list/train'
        print('Using Train data only')
    elif data == 'train_val':
        file_list = 'list/train_val'
        print('Using Train & Val data')

    file_read = open(file_list, "r")
    X_df = []
    y_df = []

    for line in file_read.readlines():
        video_id, label = line.split(' ')
        label = label.replace('\n','')

        if feat_dir == 'all/':
            # Set the feature type (both surf and cnn) and directory
            file_path_s = 'kmeans_surf/' + video_id + '.feature'
            file_path_c = 'kmeans_cnn/' + video_id + '.feature'

            if os.path.exists(file_path_s) is False:
                continue
            else:
                X_s = np.loadtxt(file_path_s, dtype=float)
                Y_label = label

            if os.path.exists(file_path_c) is False:
                continue
            else:
                X_c = np.loadtxt(file_path_c, dtype=float)
                Y_label = label

            if Y_label == event_name:
                y = 1
    #        elif Y_label == 'NULL':
    #            y = -1
            else:
                y = 0

            X = np.hstack([X_s, X_c])
            X_df.append(X)
            y_df.append(y)

        elif feat_dir == 'kmeans_surf/' or feat_dir == 'kmeans_cnn/': 
            # Set the feature type (both surf and cnn) and directory
            file_path = feat_dir + video_id + '.feature'

            if os.path.exists(file_path) is False:
                continue
            else:
                X = np.loadtxt(file_path, dtype=float)
                Y_label = label

            if Y_label == event_name:
                y = 1
    #        elif Y_label == 'NULL':
    #            y = -1
            else:
                y = 0

            X_df.append(X)
            y_df.append(y)

        elif feat_dir == 'cnn/':
            # Set the feature type (both surf and cnn) and directory
            file_path = feat_dir + video_id + '.pkl'

            if os.path.exists(file_path) is False:
                continue
            else:
                #X = np.loadtxt(file_path, dtype=float)
                with open(file_path, 'rb') as f:
                    X = pickle.load(f)   
                Y_label = label

            if Y_label == event_name:
                y = 1
    #        elif Y_label == 'NULL':
    #            y = -1
            else:
                y = 0

            X_df.append(X)
            y_df.append([y]*X.shape[0])

    file_read.close() 
    
    # Highly imbalanced data for training
    if feat_dir == 'cnn/':
        X_df = np.vstack(X_df)
        y_df = np.hstack(y_df)

        #Random sampling
        df = np.hstack([X_df, y_df.reshape(-1,1)])
        np.random.shuffle(df)
        X_df = df[:int(df.shape[0]/5),:-1]
        y_df = df[:int(df.shape[0]/5),-1]
    else: 
        X_df = np.array(X_df)
        y_df = np.array(y_df)

    # Add SMOTE resampling for dealing with imbalanced issue
    smote = SMOTE()
    X_df_res, y_df_res = smote.fit_sample(X_df, y_df)
    
    # Train lgbm
    clf = LGBMClassifier(n_estimators=500, class_weight='balanced', metric = 'binary_error', objective = 'binary', random_state=0)
    clf.fit(X_df_res, y_df_res)    

    # Save the LGBM model
    with open(output_file, 'wb') as f:
        pickle.dump(clf, f)    

    print ('LGBM trained successfully for event {}!'.format(event_name))            