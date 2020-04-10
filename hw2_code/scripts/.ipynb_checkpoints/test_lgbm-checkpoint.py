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
import tqdm

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass
warnings.warn = warn

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print ("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print ("model_file -- path of the trained svm file")
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features; provided just for debugging")
        print ("data -- train only or train_val (train+val)")
        print ("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    data = sys.argv[4]
    output_file = sys.argv[5]

    
    # Load the svm model
    svm = pickle.load(open(model_file, 'rb'))

    #file list for val or test
    if data == 'train':
        file_list = '/home/ubuntu/11775-hws/all_val.lst'
        print('Using Train data only')
    elif data == 'train_val':
        file_list = '/home/ubuntu/11775-hws/all_test_fake.lst'
        print('Using Train & Val data')
    
    
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
            file_path_s = 'kmeans_surf/' + video_id + '.feature'
            if os.path.exists(file_path_k) is False:
                X_df_s.append(np.array([0]*180)) # The number of kmeans features
            else:
                X = np.loadtxt(file_path_k)
                X_df_s.append(np.array(X))

            file_path_c = 'kmeans_cnn/' + video_id + '.feature'
            if os.path.exists(file_path_a) is False:
                X_df_c.append(np.array([0]*8006)) # The number of asrs features
            else:
                X = np.loadtxt(file_path_a)
                X_df_c.append(np.array(X))

        X_df_k = np.array(X_df_s)
        X_df_a = np.array(X_df_c)
        X_df = np.concatenate((X_df_s, X_df_c), axis = 1)
        X_df = np.array(X_df)

    elif feat_dir == 'kmeans_surf/' or feat_dir == 'kmeans_cnn/':
        X_df = []
        for video_id in video_ids:
            file_path = feat_dir + video_id + '.feature'
            if os.path.exists(file_path) is False:
                X_df.append(np.array([0]*feat_dim))
            else:
                X = np.loadtxt(file_path)
                X_df.append(np.array(X))

        X_df = np.array(X_df)

    elif feat_dir == 'cnn/':
        X_df = []
        for video_id in video_ids:
            file_path = feat_dir + video_id + '.pkl'
            if os.path.exists(file_path) is False:
                X_df.append(np.array([0]*feat_dim))
            else:
                #X = np.loadtxt(file_path)
                with open(file_path, 'rb') as f:
                    X = pickle.load(f)   
                Y_label = label
                X_df.append(np.array(X))

        X_df = np.array(X_df)

    # Save Prediction
    if feat_dir == 'cnn/':
        y_pred_proba = []
        for x in tqdm.tqdm(X_df):
            y_pred_proba_x = np.mean(svm.predict_proba(x)[:,1])
            y_pred_proba.append(y_pred_proba_x)
            svm.classes_
            np.savetxt(output_file, y_pred_proba)        
    else:
        y_pred_proba = svm.predict_proba(X_df)
        svm.classes_
        np.savetxt(output_file, y_pred_proba[:,1])
        
    if data == 'train':
        #Check the result of prediction
        if feat_dir == 'cnn/':
            y_pred = []
            for x in tqdm.tqdm(X_df):
                y_pred_x = svm.predict(X_df[0])
                y_pred_x = Counter(y_pred_x).most_common()[0][0]
                y_pred.append(y_pred_x)
        else:
            y_pred = svm.predict(X_df)
        target_event = model_file.split('.')[1]

        labels_idx = np.array(labels)
        for i in range(labels_idx.shape[0]):
            if labels_idx[i] == target_event:
                labels_idx[i] = 1
            else:
                labels_idx[i] = 0

        labels_idx = labels_idx.astype(int)

        predict_accuracy = accuracy_score(labels_idx, y_pred)
        predict_precision = average_precision_score(labels_idx, y_pred)

        print ('  LGBM prediction for event {} !'.format(target_event))
        print ('  LGBM prediction accuracy {}'.format(predict_accuracy))
        print ('  LGBM prediction precision {}'.format(predict_precision))  
        
    elif data == 'train_val':
        target_event = model_file.split('.')[1]
        print ('  LGBM prediction for event {} !'.format(target_event))
        print ('  LGBM prediction for test file saved')
        
        