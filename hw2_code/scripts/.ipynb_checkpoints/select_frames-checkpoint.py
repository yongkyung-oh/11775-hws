#!/bin/python
# Randomly select 

import numpy as np
from sklearn.preprocessing import normalize
import os
import sys
import pickle
import tqdm

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} file_list select_ratio output_file".format(sys.argv[0]))
        print ("file_list -- the list of video names")
        print ("select_ratio -- the ratio of frames to be randomly selected from each file")
        print ("output_file -- path to save the selected frames (feature vectors)")
        exit(1)

    file_list = sys.argv[1]; 
    output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    np.random.seed(18877)
    
    if output_file.split('.')[1] == 'surf':
        feature_name = 'surf'
        print('SURF features')
    elif output_file.split('.')[1] == 'cnn':
        feature_name = 'cnn'
        print('CNN features')
    else:
        raise(ValueError('Invalid data'))    
        
    for line in tqdm.tqdm(fread.readlines()):
        file_name = str(line.replace('\n','') + '.pkl')
        file_path = os.path.join(feature_name, file_name)

        #print(file_path)
        if os.path.exists(file_path) == False:
            print('{} not exist'.format(file_name))
            continue

        with open(file_path, 'rb') as f:
            data = pickle.load(f)   

        select_size = int(len(data) * ratio)
        np.random.shuffle(data)

        for n in range(select_size):
            if feature_name == 'surf':
                feature = data[n].squeeze()
                if feature.shape ==():
                    continue
                feature = feature.reshape(-1,64)
                feature = normalize(feature, axis = 1)
                #feature = feature.flatten()
                #Feature lengths vary. Select 100 per one video.
                feat_len = feature.shape[0]
                feat_dim = feature.shape[1]

                for f in range(0, min(feat_len,100)):
                    val = str(feature[f][0])
                    for m in range(1, feat_dim):
                        val += ';' + str(feature[f][m])
                    fwrite.write(val + '\n')
            elif feature_name == 'cnn':
                feature = data[n]
                feature = feature.reshape(-1,512)
                feature = normalize(feature, axis = 1)
                #feature = feature.flatten()
                #Feature lengths vary. 
                feat_dim = feature.shape[1]

                val = str(feature[0][0])
                for m in range(1, feat_dim):
                    val += ';' + str(feature[0][m])
                fwrite.write(val + '\n')
    fwrite.close()        
