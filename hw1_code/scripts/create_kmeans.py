#!/bin/python

import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster.k_means_ import MiniBatchKMeans
import cPickle
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model, 'r'))

    file_read = open(file_list, "r")
    X_shapes = []

    for line in file_read.readlines():
        video_id = line.replace('\n', '')
        mfcc_path = "mfcc/" + video_id + ".mfcc.csv"

        # Check the availability of the mfcc file
        if os.path.exists(mfcc_path) is False:
            continue

        # Read data
        X = np.loadtxt(mfcc_path, delimiter = ';')
        #X = np.genfromtxt(mfcc_csv_file, delimiter=";")
        #print (np.shape(X))
        X_shapes.append(np.array(X).shape)
        # Predict cluster
        y = kmeans.predict(X)
        y_count = np.bincount(y, minlength = cluster_num)

        # Normalize the value
        y_norm = y_count.astype(float) / cluster_num
        y_norm = y_norm / sum(y_norm)

        # Save the feature of MFCC
        np.savetxt("kmeans/" + video_id + ".feature", y_norm.transpose())    

    file_read.close()

    X_shapes = np.array(X_shapes)
    print(sum(X_shapes[:,0]))
    print(X_shapes[0,1])
    
    print "K-means features generated successfully!"