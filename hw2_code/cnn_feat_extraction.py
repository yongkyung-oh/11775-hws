#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#https://github.com/lukemelas/EfficientNet-PyTorch
#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b0')
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential()
#model.eval()

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

def get_features(model, tfms, image_pil):
    #img = tfms(Image.open(image)).unsqueeze(0)
    img = tfms(image_pil).unsqueeze(0)
    #with torch.no_grad():
    features = model(img)
    features = features.squeeze(0).detach()
    return features.numpy()

def get_cnn_features_from_video(downsampled_video_filename, video_name, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."

    images = get_keyframes(downsampled_video_filename,keyframe_interval)
    data = []
    counter = 0
    for image in images:
        image_cv = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv)
#        cv2.imwrite('frame.jpg', image)
        features = get_features(model, tfms, image_pil)
        data.append(features)
    #print(downsampled_video_filename)
    try:
        data = np.array(data)
    except:
        #continue
        pass
        
    #if data is not []:
    pickle.dump(data, open(str('cnn/'+video_name+'.pkl'), 'wb'))

        
def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    #hessian_threshold = my_params.get('hessian_threshold')
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # Check if folder for CNN features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)

    fread = open(all_video_names, "r")
    for line in tqdm.tqdm(fread.readlines()):
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')

        if not os.path.isfile(downsampled_video_filename):
            continue

        if os.path.isfile(str('cnn/'+video_name+'.pkl')):
           #print('{} exists'.format(str('cnn/'+video_name+'.pkl')))
           continue

        # Get CNN features for one video
        get_cnn_features_from_video(downsampled_video_filename,
                                     video_name, keyframe_interval)
