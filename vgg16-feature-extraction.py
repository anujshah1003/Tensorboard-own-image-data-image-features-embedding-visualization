# -*- coding: utf-8 -*-

import os, argparse
import pickle
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras import backend as K
import keras
    
# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the VGG_model/README.md to download VGG weights
CNN_weights_file_name   = 'VGG_model/vgg16_weights.h5'
image_file_name = 'elephant.jpg'
question = u'What animal is in the picture?'


# Chagne the value of verbose to 0 to avoid printing the progress statements
verbose = 1

def get_image_model(CNN_weights_file_name):
    ''' Takes the CNN weights file, and returns the VGG model update 
    with the weights. Requires the file VGG.py inside models/CNN '''
    from VGG_model.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)

    # this is standard VGG 16 without the last two layers
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may experiment with "adam" optimizer, but the loss function for
    # this kind of task is pretty standard
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

vgg16_model = get_image_model(CNN_weights_file_name)

def get_image_features(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))


    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False) # shape of im = (224,224,3)
    
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]        

    im = im.transpose((2,0,1)) # convert the image to RGBA  # shame of im= (3,224,224)

    
    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)  # shape of im = (1,3,224,224)

    image_features[0,:] = vgg16_model.predict(im)[0]
    return image_features


PATH=os.getcwd()
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

image_features_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Extracting Features of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        image_features=get_image_features(data_path + '/'+ dataset + '/'+ img )
        image_features_list.append(image_features)
    
    
image_features_arr=np.asarray(image_features_list)
image_features_arr = np.rollaxis(image_features_arr,1,0)
image_features_arr = image_features_arr[0,:,:]

np.savetxt('feature_vectors_400_samples.txt',image_features_arr)
#feature_vectors = np.loadtxt('feature_vectors.txt')
pickle.dump(image_features_arr, open('feature_vectors_400_samples.pkl', 'wb'))

        


