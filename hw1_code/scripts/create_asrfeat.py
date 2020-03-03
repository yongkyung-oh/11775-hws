#!/bin/python
import numpy as np
import os
import cPickle
from nltk.tokenize import RegexpTokenizer
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)

    vocab_file = sys.argv[1]
    file_list = sys.argv[2]

    asr_file_list = '/home/ubuntu/11775-hws/all_asr.lst'
    asr_file_read = open(asr_file_list, "r")
    
    # Generate vocab flie from the asrs file
    vocab = dict()

    tokenizer = RegexpTokenizer(r'\w+')

    n = 0
    for line in asr_file_read.readlines():
        video_id = line.replace('\n', '')[:-4]
        asr_path = "/home/ubuntu/asrs/" + video_id + ".txt"

        # Check the availability of the ars file
        # Ignore ctm file, only use txt file
        if os.path.exists(asr_path) is False:
            continue

    #    print(asr_path)

        with open(asr_path, 'r') as f:
            text = f.readlines()
            text = [tokenizer.tokenize(t) for t in text]

            for words in text:
                for word in words:
                    if word not in vocab:
                        vocab[word] = n
                        n = n + 1
            f.close()

    vocab_sorted = dict()
    sorted_key = sorted(vocab.keys(), key=lambda x:x.lower())

    n=0
    with open('vocab', 'w') as f:
        for w in sorted_key:
            vocab_sorted[w] = n
            n = n + 1
            f.write(w+'\n')
            f.close
    print('Save the vocab with size {}'.format(n))


    # Read Vocab with sorted index
    vocab = np.loadtxt(vocab_file, dtype=str)
    vocab_sorted = dict()
    vocab_size = len(vocab)
    vocab_df = [0]*vocab_size
    for n, w in enumerate(vocab):
        vocab_sorted[w] = n

    file_read = open(file_list, "r")

    for line in file_read.readlines():
        video_id = line.replace('\n', '')
        asr_path = "/home/ubuntu/asrs/" + video_id + ".txt"

        # Check the availability of the ars file
        # Ignore ctm file, only use txt file
        if os.path.exists(asr_path) is False:
            continue

#        print(asr_path)
        with open(asr_path, 'r') as f:
            text = f.readlines()
            text = [tokenizer.tokenize(t) for t in text]

            wordset = []
            for words in text:
                for word in words:
                    word.replace("'", "")
                    word.replace("_", "")
                    wordset.append(word)
            f.close()

        X = np.zeros(vocab_size)
        for word in wordset:
            X[vocab_sorted[word]] += 1

        if np.sum(X) != 0:
            X_norm = X.astype(float) / np.sum(X)

        np.savetxt("asrfeat/" + video_id + ".feature", X_norm.transpose())

    file_read.close()
        
    print "ASR features generated successfully!"