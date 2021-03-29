#####
# Image (semantic) segmentation
# For each pixel in a test image, predict the class of the object containing that pixel or
# `background' if the pixel does not belong to one of the 20 specified classes.

# Dataset: Pascal VOC
# A dataset that can be used for classification, segmentation, detection, and action classification.
# There are 20 object classes.

# Model: U-net

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# Image (semantic) segmentation
# ref.
# https://www.tensorflow.org/tutorials/images/segmentation
# https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb
# It aims to segment the image, i.e. clustering parts of an image together which belong to the same object.
# In other words, semantic segmentation is a image classification at pixel level (thus, localization is important).
# It outputs a pixel-wise mask of the image = labels for each pixel of the image with a category label; e.g.:
#   Class 1 : Pixel belonging to an object. ( 20 objects in Pascal VOC)
#   Class 2 : Pixel bordering the object. (Not applicable in Pascal VOC)
#   Class 3 : None of the above/ Surrounding pixel. (predict total 21 objects in Pascal VOC)

# Fully Convolution Network (FCN)

# U-Net
# ref. U-Net: Convolutional Networks for Biomedical Image Segmentation
# Architecture
#   (1) Contracting path (left side of U-shape; encoder; typical convolutional network)
#       - Max-pooling: increase the number of feature channels to propagate
#       - Down-sampling: take high resolution features (context information) from the image, which are cropped and
#         combined with up-sampling output to keep accurate localization information
#   (2) Expansive path (right side of U-shape; decoder; symmetric to the contracting path)
#       - Up-sampling: increase the resolution of the output, combined with the cropped high resolution feature
#         from the contracting path to localize
#   (3) (no FC layers): the segmentation map only contains the pixel
# Other techniques used
#   - Data augmentation: a technique to increase the diversity of the training set by applying transformations such
#     as image rotation; here, we use elastic deformation to efficiently train with very few annotated images
#       - Elastic deformation: a change in shape of a material at low stress that is recoverable after the stress
#         is removed; sort of skewing the pixel value
#         ref. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis
#         *** operation?
#   - Overlap tile: fill the missing context of the border region with the mirrored input image.
#   - Weighted loss: designed to tackle imbalanced data in back/foreground classification;
#     up-weight the mis-classification error of less frequent classes in the cross-entropy loss

# Import necessary libraries
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate

# Set the value of hyper-parameters
ori_size = (216, 216)
#resize_size = (72, 72)   # 원본크기가 너무 크니까 줄여주기
learning_rate = 0.05     # for u-net, start with larger learning rate
batch_size = 16
epochs = 20

# Results
# ==> - loss: 1.9307 - accuracy: 0.5566

# Load the dataset
# 1. Download the Pascal VOC dataset and unzip the tar file:
r'''
# download the Pascal VOC dataset (cmd)
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# unzip the tar file (1) (cmd)
tar xf VOCtrainval_11-May-2012.tar
# OR (2) (python)
import tarfile
import mxnet  # may need to install the package (run on cmd: pip install mxnet)

base_dir = C:\Users\yunji\VOCdevkit  # this may differ from your folder location
tar_dir = C:\Users\yunji
fp = tarfile.open(tar_dir, 'r')
fp.extractall(base_dir)  # extract the tar file
'''
#   2. Get the path to the folder where the dataset exists
voc_dir = 'C:/Users/yunji' + '/VOCdevkit/VOC2012'

#   3. Understand the directory structure
#       + VOCdevkit
#           + VOC2012
#               + Annotations: annotations for object detection; xml files
#               + ImageSets: list of image file name classified by classes or train/trainval/val; txt files
#               + JPEGImages: input images; jpg files
#               + SegmentationClass: segmentation label (mask) by class (semantic); png files
#               + SegmentationObject: segmentation label (mask) by object (instance); png files

#   4. get image and label for semantic segmentation

# With PIL library (PIL: Python Imaging Library)
def read_voc_images(voc_dir, ori_size, is_train=True):
    """Read all VOC feature and label images."""
    img_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')    # os.path.join(): 경로를 병합하여 새 경로 생성
                                                                                                              # 새 경로: C:\Users\yunji\VOCdevkit\VOC2012\ImageSets\Segmentation\train.txt or val.txt
    with open(img_fname, 'r') as f:                                                                           # 'r'은 읽기모드: 파일을 읽기만 할 때 사용
        images = f.read().split()                                                                             # 파일 읽기 -> split: 문자열을 리스트로 받는 방법

    xs, ys, ys_w255, ys_wo255 = [], [], [], []
    
    for i, fname in enumerate(images):
        x = Image.open(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg'))                                   # read each image one by one
        y = Image.open(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'))
        
        # crop the image by trimming on all four sides and preserving the center of the image (가운데 부분만 crop 해주기 -> Data aug.중 하나)
        x = crop_center(x, ori_size)
        y = crop_center(y, ori_size)

        # Normalize
        # Converting pil.image object to np.array would remove the colormap on the ground truth annotations
        # Thus, the codes below result:
        # (1) an array of pixel values with [0:20] = class labels incl. background and 255 = border region
        # (2) pixel values multiplied by 12 for visual purpose
        # (3) a mask image in grey scale

        # y_trial = np.asarray(y) * 12                                # (1)-(2)
        # img_y = Image.fromarray(y_trial, 'L'); img_y.show()         # (3) -> 'L'의 의미=8-bit pixels, black and white; ref.https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes

        x = np.asarray(x, dtype='float32') / 255.0                    # np.asarray(): convert the input to an array, the data-type is inferred from the input data
        y = np.asarray(y, dtype='float32')
        
        y_w255, y_wo255 = y.copy(), y.copy()                          # y.copy(): y array를 그대로 복사 / y_w255: with border, y_wo255: without border
        #y_w255[y_w255 == 255] = 21                                   # 이 부분 안 넣어주면 클래스가 255개 있는걸로 인식함. 원래 20개의 class에서 하나를 추가해주는 식으로 코딩
        y_wo255[y_wo255 == 255] = 0                                   # background처럼 인식될 수 있게 0으로 값 정해줌
        #y_w255 = tf.keras.utils.to_categorical(y_w255, 22)
        y_wo255 = tf.keras.utils.to_categorical(y_wo255, 21)          # input 데이터를 one-hot encoding 해주는 과정
        
        xs.append(x)
        #ys.append(y)
        #ys_w255.append(y_w255)
        ys_wo255.append(y_wo255)
        
    x_np = np.asarray(xs)
    #y_np = np.asarray(ys)
    #y_np_w255 = np.asarray(ys_w255)
    y_np_wo255 = np.asarray(ys_wo255)
    
    return x_np, y_np_wo255 #y_np, y_np_w255, y_np_wo255

def crop_center(pil_img, ori_size):
    """ Crop the image to the given size by trimming on all four sides and preserving the center of the image. """
    (crop_width, crop_height) = ori_size
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


x_train, y_train_wo255 = read_voc_images(voc_dir, ori_size, True) # y_train, y_train_w255,
x_valid, y_valid_wo255 = read_voc_images(voc_dir, ori_size, False) # y_valid, y_valid_w255,

# Preprocess the dataset
# Data-type conversion
# Normalization
# One-hot encoding


# Build a U-Net architecture
# Input
input_tensor = Input(shape=ori_size + (3,), name='input_tensor') #216,216,3

# Contraction path
cont1_1 = Conv2D(64, 3, activation='relu', padding='same', name='cont1_1')(input_tensor) #216,216,64
cont1_2 = Conv2D(64, 3, activation='relu', padding='same', name='cont1_2')(cont1_1) #216,216,64

cont2_dwn = MaxPooling2D((2, 2), strides=2, name='cont2_dwn')(cont1_2) #108,108,64
cont2_1 = Conv2D(128, 3, activation='relu', padding='same', name='cont2_1')(cont2_dwn) #108,108,128
cont2_2 = Conv2D(128, 3, activation='relu', padding='same', name='cont2_2')(cont2_1) #108,108,128

cont3_dwn = MaxPooling2D((2, 2), strides=2, name='cont3_dwn')(cont2_2) #54,54,128
cont3_1 = Conv2D(256, 3, activation='relu', padding='same', name='cont3_1')(cont3_dwn) #54,54,256
cont3_2 = Conv2D(256, 3, activation='relu', padding='same', name='cont3_2')(cont3_1) #54,54,256

cont4_dwn = MaxPooling2D((2, 2), strides=2, name='cont4_dwn')(cont3_2) #27,27,256
cont4_1 = Conv2D(512, 3, activation='relu', padding='same', name='cont4_1')(cont4_dwn) #27,27,512
cont4_2 = Conv2D(512, 3, activation='relu', padding='same', name='cont4_2')(cont4_1) #27,27,512

r'''
# reduce the depth of u-net
cont5_dwn = MaxPooling2D((2, 2), strides=2, name='cont5_dwn')(cont4_2)  # down-sampling
cont5_1 = Conv2D(1024, 3, activation='relu', padding='same', name='cont5_1')(cont5_dwn)
cont5_2 = Conv2D(1024, 3, activation='relu', padding='same', name='cont5_2')(cont5_1)
'''

# Expansive path
# *** UpSampling2D vs. Conv2DTranspose:
# ref. https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker

r'''
# reduce the depth of network
expn1_up = Conv2DTranspose(512, 2, strides=2, name='expn1_up')(cont5_2)  # up-sampling
cropping_size = (cont4_2.shape[1] - expn1_up.shape[1]) // 2
cropping = ((cropping_size, cropping_size), (cropping_size, cropping_size))
expn1_crop = Cropping2D(cropping, name='expn1_crop')(cont4_2)  
expn1_concat = concatenate([expn1_up, expn1_crop], axis=-1, name='expn1_concat')  
expn1_1 = Conv2D(512, 3, activation='relu', padding='same', name='expn1_1')(expn1_concat)  
expn1_2 = Conv2D(512, 3, activation='relu', padding='same', name='expn1_2')(expn1_1)
'''

# Apply activation first, then concat
expn2_up = Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer='he_normal', name='expn2_up')(cont4_2) #54,54,256
expn2_concat = concatenate([expn2_up, cont3_2], axis=-1, name='expn2_concat') #54,54,512
expn2_1 = Conv2D(256, 3, activation='relu', padding='same', name='expn2_1')(expn2_concat) #54,54,256
expn2_2 = Conv2D(256, 3, activation='relu', padding='same', name='expn2_2')(expn2_1) #54,54,256

expn3_up = Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer='he_normal', name='expn3_up')(expn2_2) #108,108,128
expn3_concat = concatenate([expn3_up, cont2_2], axis=-1, name='expn3_concat') #108,108,256
expn3_1 = Conv2D(128, 3, activation='relu', padding='same', name='expn3_1')(expn3_concat) #108,108,128
expn3_2 = Conv2D(128, 3, activation='relu', padding='same', name='expn3_2')(expn3_1) #108,108,128

expn4_up = Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer='he_normal', name='expn4_up')(expn3_2) #216,216,64
expn4_concat = concatenate([expn4_up, cont1_2], axis=-1, name='expn4_concat') #216,216,128
expn4_1 = Conv2D(64, 3, activation='relu', padding='same', name='expn4_1')(expn4_concat) #216,216,64
expn4_2 = Conv2D(64, 3, activation='relu', padding='same', name='expn4_2')(expn4_1) #216,216,64

# one hot w/o 255
output_tensor = Conv2D(20 + 1, 1, padding='same', activation='softmax', name='output_tensor')(expn4_2) #216,216,21

# Create a model
u_net = Model(input_tensor, output_tensor, name='u-net')
u_net.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
u_net.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
u_net.fit(x_train, y_train_wo255, batch_size=batch_size, epochs=epochs)

# Test the model with test set
u_net.evaluate(x_valid, y_valid_wo255, verbose=1)
#imgs = u_net.predict(x_valid)
#img = Image.fromarray(imgs[0], 'RGB')
#img.show()

'''
#import matplotlib.pyplot as plt
#imgs = u_net.predict(x_valid[:10])

#plt.imshow(x_valid[0])
#plt.imshow(tf.argmax(y_valid_wo255[0], axis=-1))
#plt.imshow(tf.argmax(imgs[0], axis=-1))
'''