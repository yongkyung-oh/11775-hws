#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
map_path=/home/ubuntu/tools/mAP
export PATH=$map_path:$PATH


echo "#####################################"
echo "#       MED with MFCC Features      #"
echo "#####################################"
mkdir -p submit_pred
# iterate over the events
feat_dim_mfcc=180
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_lgbm.py $event "kmeans/" $feat_dim_mfcc submit_pred/lgbm.$event.model False || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_lgbm_all.py submit_pred/lgbm.$event.model "kmeans/" $feat_dim_mfcc submit_pred/${event}_mfcc_lgbm_all_test_fake.lst || exit 1;
  # compute the average precision by calling the mAP package
#  ap list/${event}_val_label mfcc_pred/${event}_mfcc.lst
done

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p submit_pred
# iterate over the events
feat_dim_asr=8006
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_lgbm.py $event "asrfeat/" $feat_dim_asr submit_pred/lgbm.$event.model False || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_lgbm_all.py submit_pred/lgbm.$event.model "asrfeat/" $feat_dim_asr submit_pred/${event}_asr_lgbm_all_test_fake.lst || exit 1;
  # compute the average precision by calling the mAP package
#  ap list/${event}_val_label asr_pred/${event}_asr.lst
done

echo ""
echo "#####################################"
echo "#       MED with ALL Features       #"
echo "#####################################"
mkdir -p submit_pred
# iterate over the events
feat_dim_all=8186
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_lgbm.py $event "all/" $feat_dim_all submit_pred/lgbm.$event.model False || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_lgbm_all.py submit_pred/lgbm.$event.model "all/" $feat_dim_all submit_pred/${event}_all_lgbm_all_test_fake.lst || exit 1;
  # compute the average precision by calling the mAP package
#  ap list/${event}_val_label asr_pred/${event}_asr.lst
done
