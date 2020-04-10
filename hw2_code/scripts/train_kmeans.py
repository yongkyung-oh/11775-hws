#!/bin/python 

import numpy as np
from sklearn.preprocessing import normalize
import os
import glob
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print ("feature_file -- path to the feature files")
        print ("cluster_num -- number of cluster")
        print ("output_file -- path to save the k-means model")
        exit(1)
        
    feature_file = sys.argv[1]
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    if feature_file.split('.')[1] == 'surf':
        feature_name = 'surf'
        print('SURF features')
    elif feature_file.split('.')[1] == 'cnn':
        feature_name = 'cnn'
        print('CNN features')
    else:
        raise(ValueError('Invalid data'))
        
    # Read data
    X = np.loadtxt(feature_file, delimiter = ';')
    #X = np.genfromtxt(mfcc_csv_file, delimiter=";")
    print (X.shape)
    
    X = normalize(X, axis = 1)
    # Model fit to data
    #kmeans = KMeans(n_clusters=cluster_num)
    kmeans = MiniBatchKMeans(n_clusters=cluster_num, batch_size = 50, random_state=0, init_size = 500) #Convential KMeans is too slow #Potential Memory Error
    kmeans.fit(X)

    # Save model
    pickle.dump(kmeans, open(output_file, 'wb'))

    print ("K-means trained successfully!")