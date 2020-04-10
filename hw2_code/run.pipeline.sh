#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -y filepath

# Reading of all arguments:
while getopts p:f:m:y: option		# p:f:m:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

#export PATH=~/anaconda3/bin:$PATH
export PATH=~/anaconda3/envs/LSMM/:$PATH


if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=~/videos  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans_surf kmeans_cnn  # create folders to save features
    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../hw1_code/list/val > list/val.video
    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/all.video"); do
        if test -f downsampled_videos/$line.ds.mp4; then
          #echo "$file_name exist"
          continue
        else        
          echo "$file_name not exist"
          ffmpeg -threads 2 -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
        fi
    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    #python surf_feat_extraction.py -i list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
    python cnn_feat_extraction.py list/all.video config.yaml	

    echo "PREPROCESS SUCCESSFUL COMPLETION"
fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then
    #cluster_num=180        # the number of clusters in k-means.
    cluster_num=360        # the number of clusters in k-means.

    mkdir -p kmeans_surf kmeans_cnn  # create folders to save features

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    echo "Pooling SURFs (20%)"
    #python scripts/select_frames.py list/train.video 0.1 select.surf.10.csv || exit 1;
    #python scripts/select_frames.py list/train.video 0.2 select.surf.20.csv || exit 1;

    echo "Training the k-means model"
    #python scripts/train_kmeans.py select.surf.20.csv $cluster_num kmeans.surf.${cluster_num}.model || exit 1;

    # 2. TODO: Create kmeans representation for SURF features
    echo "Creating k-means cluster vectors"
    #python scripts/create_kmeans.py kmeans.surf.${cluster_num}.model $cluster_num list/all.video || exit 1;

	  echo "#####################################"
    echo "#   CNN FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for CNN features
    echo "Pooling CNNs (50%)"
    #python scripts/select_frames.py list/train.video 0.5 select.cnn.50.csv || exit 1;

    echo "Training the k-means model"
    #python scripts/train_kmeans.py select.cnn.50.csv $cluster_num kmeans.cnn.${cluster_num}.model || exit 1;

    # 2. TODO: Create kmeans representation for CNN features
    echo "Creating k-means cluster vectors"
    #python scripts/create_kmeans.py kmeans.cnn.${cluster_num}.model $cluster_num list/all.video || exit 1;

    echo "FEATURE EXTRACTION SUCCESSFUL COMPLETION"
fi

if [ "$MAP" = true ] ; then
    # Paths to different tools;
    #map_path=/home/ubuntu/tools/mAP
    #export PATH=$map_path:$PATH

    #cat list/train list/val > list/train_val    #save train & val video names in one file
    
    for model in svm lgbm; do
      echo ""
      echo "=========  Model $model  ========="

      echo "########################################"
      echo "# MED with SURF Kmeans: MAP results    #"
      echo "########################################"

      mkdir -p pred_surf
      feat_dim_surf=360
      for event in P001 P002 P003; do
        echo "=========  Event $event  ========="
        # 1. TODO: Train SVM/LGBM with OVR using only videos in training set.
        python scripts/train_$model.py $event "kmeans_surf/" $feat_dim_surf "train" pred_surf/$model.$event.train.model || exit 1;
      
        # 2. TODO: Test SVM/LGBM with val set and calculate its MAP scores for own info.
        python scripts/test_$model.py pred_surf/$model.$event.train.model "kmeans_surf/" $feat_dim_surf "train" pred_surf/${event}_surf.val.$model.lst || exit 1;

        # 3. TODO: Train SVM/LGBM with OVR using videos in training and validation set.
        python scripts/train_$model.py $event "kmeans_surf/" $feat_dim_surf "train_val" pred_surf/$model.$event.train_val.model || exit 1;

        # 4. TODO: Test SVM/LGBM with test set saving scores for submission
        python scripts/test_$model.py pred_surf/$model.$event.train_val.model "kmeans_surf/" $feat_dim_surf "train_val" pred_surf/${event}_surf.test.$model.lst || exit 1;
      done
    done

    for model in svm lgbm; do
      echo ""
      echo "=========  Model $model  ========="

      echo "########################################"
      echo "# MED with CNN Kmeans: MAP results     #"
      echo "########################################"

      mkdir -p pred_cnn
      feat_dim_cnn=360
      for event in P001 P002 P003; do
        echo "=========  Event $event  ========="
        # 1. TODO: Train SVM/LGBM with OVR using only videos in training set.
        python scripts/train_$model.py $event "kmeans_cnn/" $feat_dim_cnn "train" pred_cnn/$model.$event.train.model || exit 1;
      
        # 2. TODO: Test SVM/LGBM with val set and calculate its MAP scores for own info.
        python scripts/test_$model.py pred_cnn/$model.$event.train.model "kmeans_cnn/" $feat_dim_surf "train" pred_cnn/${event}_cnn.val.$model.lst || exit 1;

        # 3. TODO: Train SVM/LGBM with OVR using videos in training and validation set.
        python scripts/train_$model.py $event "kmeans_cnn/" $feat_dim_cnn "train_val" pred_cnn/$model.$event.train_val.model || exit 1;

        # 4. TODO: Test SVM/LGBM with test set saving scores for submission
        python scripts/test_$model.py pred_cnn/$model.$event.train_val.model "kmeans_cnn/" $feat_dim_cnn "train_val" pred_cnn/${event}_cnn.test.$model.lst || exit 1;
      done
    done

    for model in svm lgbm; do
      echo ""
      echo "=========  Model $model  ========="

      echo "########################################"
      echo "# MED with CNN Features: MAP results   #"
      echo "########################################"

      mkdir -p pred_cnn_feature
      feat_dim_cnn=512
      for event in P001 P002 P003; do
        echo "=========  Event $event  ========="
        # 1. TODO: Train SVM/LGBM with OVR using only videos in training set.
        #python scripts/train_$model.py $event "cnn/" $feat_dim_cnn "train" pred_cnn_feature/$model.$event.train.model || exit 1;
      
        # 2. TODO: Test SVM/LGBM with val set and calculate its MAP scores for own info.
        #python scripts/test_$model.py pred_cnn_feature/$model.$event.train.model "cnn/" $feat_dim_surf "train" pred_cnn_feature/${event}_cnn.val.$model.lst || exit 1;

        # 3. TODO: Train SVM/LGBM with OVR using videos in training and validation set.
        #python scripts/train_$model.py $event "cnn/" $feat_dim_cnn "train_val" pred_cnn_feature/$model.$event.train_val.model || exit 1;

        # 4. TODO: Test SVM/LGBM with test set saving scores for submission
        #python scripts/test_$model.py pred_cnn_feature/$model.$event.train_val.model "cnn/" $feat_dim_cnn "train_val" pred_cnn_feature/${event}_cnn.test.$model.lst || exit 1;
      done
    done


    echo "SUCCESSFUL COMPLETION"

fi
