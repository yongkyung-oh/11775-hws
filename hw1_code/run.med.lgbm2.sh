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
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=180
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a lgbm model
  python scripts/train_lgbm.py $event "kmeans/" $feat_dim_mfcc mfcc_pred/lgbm.$event.model True || exit 1;
  # apply the lgbm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_lgbm.py mfcc_pred/lgbm.$event.model "kmeans/" $feat_dim_mfcc mfcc_pred/${event}_mfcc.lst || exit 1;
  # compute the average precision by calling the mAP package
#  ap list/${event}_val_label mfcc_pred/${event}_mfcc.lst
done

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
feat_dim_asr=8006
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a lgbm model
  python scripts/train_lgbm.py $event "asrfeat/" $feat_dim_asr asr_pred/lgbm.$event.model True || exit 1;
  # apply the lgbm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_lgbm.py asr_pred/lgbm.$event.model "asrfeat/" $feat_dim_asr asr_pred/${event}_asr.lst || exit 1;
  # compute the average precision by calling the mAP package
#  ap list/${event}_val_label asr_pred/${event}_asr.lst
done

echo ""
echo "#####################################"
echo "#       MED with ALL Features       #"
echo "#####################################"
mkdir -p all_pred
# iterate over the events
feat_dim_asr=8306
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a lgbm model
  python scripts/train_lgbm.py $event "all/" $feat_dim_asr all_pred/lgbm.$event.model True || exit 1;
  # apply the lgbm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_lgbm.py all_pred/lgbm.$event.model "all/" $feat_dim_asr all_pred/${event}_all.lst || exit 1;
  # compute the average precision by calling the mAP package
#  ap list/${event}_val_label asr_pred/${event}_asr.lst
done