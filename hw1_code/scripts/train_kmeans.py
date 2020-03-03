#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster.k_means_ import MiniBatchKMeans
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # Read data
    X = np.loadtxt(mfcc_csv_file, delimiter = ';')
    #X = np.genfromtxt(mfcc_csv_file, delimiter=";")
    print (np.shape(X))

    # Model fit to data
    #kmeans = KMeans(n_clusters=cluster_num)
    kmeans = MiniBatchKMeans(n_clusters=cluster_num, batch_size = 50, random_state=0) #Convential KMeans is too slow #Potential Memory Error
    kmeans.fit(X)

    # Save model
    cPickle.dump(kmeans, open(output_file, 'w'))
    
    print "K-means trained successfully!"
