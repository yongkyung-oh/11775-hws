#!/bin/python

import numpy as np
from sklearn.preprocessing import normalize
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle
import sys
import tqdm

# Generate VLAD features for videos; each video is represented by vector set of frames

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print ("kmeans_model -- path to the kmeans model")
        print ("cluster_num -- number of cluster")
        print ("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    if kmeans_model.split('.')[1] == 'surf':
        feature_name = 'surf'
        print('SURF features')
    elif kmeans_model.split('.')[1] == 'cnn':
        feature_name = 'cnn'
        print('CNN features')
    else:
        raise(ValueError('Invalid data'))    
        
    # load the kmeans model
    kmeans = pickle.load(open(kmeans_model, 'rb'))

    file_read = open(file_list,"r")
    X_shapes = []

    kmeans_center = kmeans.cluster_centers_
    
    for line in tqdm.tqdm(file_read.readlines()):
        video_id = line.replace('\n', '')
        file_name = str(line.replace('\n','') + '.pkl')
        file_path = os.path.join(feature_name, file_name)

        #print(file_path)
        if os.path.exists(file_path) == False:
            print('{} not exist'.format(file_name))
            #continue

        # Skip existing feature of Kmeans for surf and cnn
        if os.path.exists(str('vlad_cnn/'+video_id+'.pkl')):
            continue

        with open(file_path, 'rb') as f:
            try:
                data = pickle.load(f)   
            except:
                data = []

        X = []
        for feature in data:
            X.append(np.zeros((1,512)))
            feature = feature.reshape(-1,512)
            feature = normalize(feature, axis = 1)
            X.append(feature)

        if len(X) == 0:
            if feature_name == 'surf':
                X.append(np.zeros((1,64)))
            elif feature_name == 'cnn':
                X.append(np.zeros((1,512)))

        X = np.vstack(X)
        X_shapes.append(X.shape)

        V_list = []
        for x in X:
            V = np.add(kmeans_center, -x)
            V = V.flatten().reshape(1,-1)
            v = normalize(V)
            V_list.append(v)
        V_array = np.vstack(V_list)

        pickle.dump(V_array, open(str('vlad_cnn/'+video_id+'.pkl'), 'wb'))
    file_read.close()

    X_shapes = np.array(X_shapes)
#    print(sum(X_shapes[:,0]))
#    print(X_shapes[0,1])

    print ("VLAD features generated successfully!")
