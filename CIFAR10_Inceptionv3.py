#####
# dataset:
# 1) CIFAR-10
# 2) tinyImageNet
# All images are of size 64×64; 200 image classes
# Training = 100,000 images; Validation = 10,000 images; Test = 10,000 images.

# model: inception v3
# ref.
#   - Going deeper with convolutions (inception v1)
#   - Rethinking the Inception Architecture for Computer Vision (inception v2, v3)

# Notation
# ***: Questions or further information to check are remained
# NOTE: if the code is modified, be aware of the corresponding codes
#####

# General problem of Deep Neural Network
# ==> over-fitting & computationally expensive due to a large number of parameters
# Suggested solutions
# 1. Sparsely connected network / layer-by-layer (layer-wise) construction
#   - Opposed to fully connected network
#   - Cluster neurons with highly correlated outputs of the previous activation layer (Hebbian principle)
#   ==> Compared to dense matrix calculation (parallel computing), sparse matrix computation is less efficient
# 2. Inception architecture
#   - Intermediate step: sparsity at filter level + dense matrix computation
#   - The idea of sparsity is corresponded to a variation in the location of information in the image
#       ==> use a different size of kernels (larger one for global info; smaller for local)
#       ==> naive: [1x1 (cross-channel correlation)] + [3x3 + 5x5 (spatial correlation)] + 3x3maxpool
#   - To reduce computational cost:
#       ==> inception-v1: 1x1 + 1x1-3x3 + 1x1-5x5 + 3x3maxpool-1x1 -> filter concatenation
#       * 1x1 conv benefits: increased representational power of NN + dimension reduction (network-in-network)
#       ==> inception-v2,v3: 1x1 + 1x1-3x3 + 1x1-3x3-3x3 + 3x3maxpool-1x1 -> filter concatenation (factorization)

# RMSProp Optimizer; Factorized 7x7 convolutions; BatchNorm in the Auxillary Classifiers; Label Smoothing

# Import necessary libraries
import numpy as np
#from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, concatenate

# Set the value of hyper-parameters
upsampling_size = (3, 3)
learning_rate = 0.0001
epochs = 20
batch_size = 16

# Load the dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the dataset
# Data-type conversion
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0
# One-hot encoding
# 여기서는 생략함 -> 데이터가 많은 경우, 안해주는게 훨씬 효율적이다.
# 해주냐 안해주냐에 따라 밑에 loss 방법이 바뀜  (Sparse 또는 Categorical_crossentropy)

# One-hot encoding: label encoding = [6] --> one-hot encoding = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# advantage:
# - eliminate the possibility that categorical features are misinterpreted as having a hierarchical relationship
# - improve the model performance
# disadvantage:
# - lead to high memory consumption if num of categories is large

# Build a Inception V3 architecture
# Input
input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input') #32,32,3
# Resale image (up-sampling) for better performance
upsampling = tf.keras.layers.UpSampling2D(size=upsampling_size, name='upsampling')(input_tensor) #96,96,3

# conv1
conv1_conv = Conv2D(32, 3, strides=(2, 2), kernel_initializer='he_normal', name='conv1_conv')(upsampling) #47,47,32
# kernel_initializer: a statistical distribution or function to use for initialising the weights
#   - glorot_uniform: (default) also called Xavier uniform;
#       Draws samples from a uniform distribution within [-limit, limit], where limit = sqrt(6 / (fan_in + fan_out))
#       (fan_in is the number of input units in the weight tensor and fan_out is the number of output units).
#   - he_normal:
#       Draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in)
conv1_bn = BatchNormalization(axis=1, name='conv1_bn')(conv1_conv)
conv1_relu = Activation('relu', name='conv1_relu')(conv1_bn)

# conv2_1
conv2_1conv = Conv2D(32, 3, kernel_initializer='he_normal', name='conv2_1conv')(conv1_relu) #45,45,32
conv2_1bn = BatchNormalization(axis=1, name='conv2_1bn')(conv2_1conv)
conv2_1relu = Activation('relu', name='conv2_1relu')(conv2_1bn)

# conv2_2
conv2_2conv = Conv2D(64, 3, padding='SAME', kernel_initializer='he_normal', name='conv2_2conv')(conv2_1relu) #45,45,64
# 지수쌤은 위에 Conv2D에 padding='SAME'을 넣어줬음
# Padded라고 표기 된 convolution과, grid size 유지가 필요한 inception module을 제외하고는 padding을 사용하지 않는다
# ref. https://sike6054.github.io/blog/paper/third-post/
conv2_2bn = BatchNormalization(axis=1, name='conv2_2bn')(conv2_2conv)
conv2_2relu = Activation('relu', name='conv2_2relu')(conv2_2bn)

# maxpool1
maxpool1 = MaxPooling2D((3, 3), strides=2, name='maxpool1')(conv2_2relu) #22,22,64

# conv2_3
conv2_3conv = Conv2D(80, 1, kernel_initializer='he_normal', name='conv2_3conv')(maxpool1) #22,22,80
conv2_3bn = BatchNormalization(axis=1, name='conv2_3bn')(conv2_3conv)
conv2_3relu = Activation('relu', name='conv2_3relu')(conv2_3bn)

# conv2_4
conv2_4conv = Conv2D(192, 3, kernel_initializer='he_normal', name='conv2_4conv')(conv2_3relu) #20,20,192
conv2_4bn = BatchNormalization(axis=1, name='conv2_4bn')(conv2_4conv)
conv2_4relu = Activation('relu', name='conv2_4relu')(conv2_4bn)

# maxpool2
maxpool2 = MaxPooling2D((3, 3), strides=2, name='maxpool2')(conv2_4relu) #9,9,192

# Inception1
# 1x1
inception1_1x1conv = Conv2D(64, 1, name='inception1_1x1conv')(maxpool2) #9,9,64
inception1_1x1bn = BatchNormalization(axis=1, name='inception1_1x1bn')(inception1_1x1conv)
inception1_1x1relu = Activation('relu', name='inception1_1x1relu')(inception1_1x1bn)
# 1x1-5x5
inception1_5x5conv1 = Conv2D(48, 1, name='inception1_5x5conv1')(maxpool2) #9,9,48
inception1_5x5bn1 = BatchNormalization(axis=1, name='inception1_5x5bn1')(inception1_5x5conv1)
inception1_5x5relu1 = Activation('relu', name='inception1_5x5relu1')(inception1_5x5bn1)
inception1_5x5conv2 = Conv2D(64, 5, padding='SAME', name='inception1_5x5conv2')(inception1_5x5relu1) #9,9,64
inception1_5x5bn2 = BatchNormalization(axis=1, name='inception1_5x5bn2')(inception1_5x5conv2)
inception1_5x5relu2 = Activation('relu', name='inception1_5x5relu2')(inception1_5x5bn2)
# 1x1-3x3-3x3
inception1_3x3conv1 = Conv2D(64, 1, name='inception1_3x3conv1')(maxpool2) #9,9,64
inception1_3x3bn1 = BatchNormalization(axis=1, name='inception1_3x3bn1')(inception1_3x3conv1)
inception1_3x3relu1 = Activation('relu', name='inception1_3x3relu1')(inception1_3x3bn1)
inception1_3x3conv2 = Conv2D(96, 3, padding='SAME', name='inception1_3x3conv2')(inception1_3x3relu1) #9,9,96
inception1_3x3bn2 = BatchNormalization(axis=1, name='inception1_3x3bn2')(inception1_3x3conv2)
inception1_3x3relu2 = Activation('relu', name='inception1_3x3relu2')(inception1_3x3bn2)
inception1_3x3conv3 = Conv2D(96, 3, padding='SAME', name='inception1_3x3conv3')(inception1_3x3relu2) #9,9,96
inception1_3x3bn3 = BatchNormalization(axis=1, name='inception1_3x3bn3')(inception1_3x3conv3)
inception1_3x3relu3 = Activation('relu', name='inception1_3x3relu3')(inception1_3x3bn3)
# avgpool
inception1_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception1_avgpool')(maxpool2) #9,9,192

inception1_avgpool_conv = Conv2D(32, 1, name='inception1_avgpool_conv')(inception1_avgpool) #9,9,32
inception1_avgpool_bn = BatchNormalization(axis=1, name='inception1_avgpool_bn')(inception1_avgpool_conv)
inception1_avgpool_relu = Activation('relu', name='inception1_avgpool_relu')(inception1_avgpool_bn)
# concat
inception1_concat = concatenate([inception1_1x1relu, inception1_5x5relu2, inception1_3x3relu3, inception1_avgpool_relu],
                                axis=-1, name='inception1_concat') #9,9,256

# concatenate를 axis 축에 대하여 진행
# 1) axis=0: 가장 높은 차원을 기준으로 합쳐줌
# 2) axis=1: 두번째로 높은 차원을 기준으로 합쳐줌
# 3) axis=-1: 가장 낮은 차원을 뒤쪽에서부터 시작
# ref: https://supermemi.tistory.com/11


# Inception2
# 1x1
inception2_1x1conv = Conv2D(64, 1, name='inception2_1x1conv')(inception1_concat) #9,9,64
inception2_1x1bn = BatchNormalization(axis=1, name='inception2_1x1bn')(inception2_1x1conv)
inception2_1x1relu = Activation('relu', name='inception2_1x1relu')(inception2_1x1bn)
# 1x1-5x5
inception2_5x5conv1 = Conv2D(48, 1, name='inception2_5x5conv1')(inception1_concat) #9,9,48
inception2_5x5bn1 = BatchNormalization(axis=1, name='inception2_5x5bn1')(inception2_5x5conv1)
inception2_5x5relu1 = Activation('relu', name='inception2_5x5relu1')(inception2_5x5bn1)
inception2_5x5conv2 = Conv2D(64, 5, padding='SAME', name='inception2_5x5conv2')(inception2_5x5relu1) #9,9,64
inception2_5x5bn2 = BatchNormalization(axis=1, name='inception2_5x5bn2')(inception2_5x5conv2)
inception2_5x5relu2 = Activation('relu', name='inception2_5x5relu2')(inception2_5x5bn2)
# 1x1-3x3-3x3
inception2_3x3conv1 = Conv2D(64, 1, name='inception2_3x3conv1')(inception1_concat) #9,9,64
inception2_3x3bn1 = BatchNormalization(axis=1, name='inception2_3x3bn1')(inception2_3x3conv1)
inception2_3x3relu1 = Activation('relu', name='inception2_3x3relu1')(inception2_3x3bn1)
inception2_3x3conv2 = Conv2D(96, 3, padding='SAME', name='inception2_3x3conv2')(inception2_3x3relu1) #9,9,96
inception2_3x3bn2 = BatchNormalization(axis=1, name='inception2_3x3bn2')(inception2_3x3conv2)
inception2_3x3relu2 = Activation('relu', name='inception2_3x3relu2')(inception2_3x3bn2)
inception2_3x3conv3 = Conv2D(96, 3, padding='SAME', name='inception2_3x3conv3')(inception2_3x3relu2) #9,9,96
inception2_3x3bn3 = BatchNormalization(axis=1, name='inception2_3x3bn3')(inception2_3x3conv3)
inception2_3x3relu3 = Activation('relu', name='inception2_3x3relu3')(inception2_3x3bn3)
# avgpool
inception2_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception2_avgpool')(inception1_concat) #9,9,256

inception2_avgpool_conv = Conv2D(64, 1, name='inception2_avgpool_conv')(inception2_avgpool) #9,9,64
inception2_avgpool_bn = BatchNormalization(axis=1, name='inception2_avgpool_bn')(inception2_avgpool_conv)
inception2_avgpool_relu = Activation('relu', name='inception2_avgpool_relu')(inception2_avgpool_bn)
# concat
inception2_concat = concatenate([inception2_1x1relu, inception2_5x5relu2, inception2_3x3relu3, inception2_avgpool_relu],
                                axis=-1, name='inception2_concat') #9,9,288

# Inception3
# 1x1
inception3_1x1conv = Conv2D(64, 1, name='inception3_1x1conv')(inception2_concat) #9,9,64
inception3_1x1bn = BatchNormalization(axis=1, name='inception3_1x1bn')(inception3_1x1conv)
inception3_1x1relu = Activation('relu', name='inception3_1x1relu')(inception3_1x1bn)
# 1x1-5x5
inception3_5x5conv1 = Conv2D(48, 1, name='inception3_5x5conv1')(inception2_concat) #9,9,48
inception3_5x5bn1 = BatchNormalization(axis=1, name='inception3_5x5bn1')(inception3_5x5conv1)
inception3_5x5relu1 = Activation('relu', name='inception3_5x5relu1')(inception3_5x5bn1)
inception3_5x5conv2 = Conv2D(64, 5, padding='SAME', name='inception3_5x5conv2')(inception3_5x5relu1) #9,9,64
inception3_5x5bn2 = BatchNormalization(axis=1, name='inception3_5x5bn2')(inception3_5x5conv2)
inception3_5x5relu2 = Activation('relu', name='inception3_5x5relu2')(inception3_5x5bn2)
# 1x1-3x3-3x3
inception3_3x3conv1 = Conv2D(64, 1, name='inception3_3x3conv1')(inception2_concat) #
inception3_3x3bn1 = BatchNormalization(axis=1, name='inception3_3x3bn1')(inception3_3x3conv1)
inception3_3x3relu1 = Activation('relu', name='inception3_3x3relu1')(inception3_3x3bn1)
inception3_3x3conv2 = Conv2D(96, 3, padding='SAME', name='inception3_3x3conv2')(inception3_3x3relu1) #9,9,96
inception3_3x3bn2 = BatchNormalization(axis=1, name='inception3_3x3bn2')(inception3_3x3conv2)
inception3_3x3relu2 = Activation('relu', name='inception3_3x3relu2')(inception3_3x3bn2)
inception3_3x3conv3 = Conv2D(96, 3, padding='SAME', name='inception3_3x3conv3')(inception3_3x3relu2) #9,9,96
inception3_3x3bn3 = BatchNormalization(axis=1, name='inception3_3x3bn3')(inception3_3x3conv3)
inception3_3x3relu3 = Activation('relu', name='inception3_3x3relu3')(inception3_3x3bn3)
# avgpool
inception3_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception3_avgpool')(inception2_concat) #9,9,288

inception3_avgpool_conv = Conv2D(64, 1, name='inception3_avgpool_conv')(inception3_avgpool) #9,9,64
inception3_avgpool_bn = BatchNormalization(axis=1, name='inception3_avgpool_bn')(inception3_avgpool_conv)
inception3_avgpool_relu = Activation('relu', name='inception3_avgpool_relu')(inception3_avgpool_bn)
# concat
inception3_concat = concatenate([inception3_1x1relu, inception3_5x5relu2, inception3_3x3relu3, inception3_avgpool_relu],
                                axis=-1, name='inception3_concat') #9,9,288

# Inception4
# different structure compared to inception3
# 3x3
inception4_3x3conv = Conv2D(384, 3, strides=(2, 2), name='inception4_3x3conv')(inception3_concat) #4,4,384
inception4_3x3bn = BatchNormalization(axis=1, name='inception4_3x3bn')(inception4_3x3conv)
inception4_3x3relu = Activation('relu', name='inception4_3x3relu')(inception4_3x3bn)
# 1x1-3x3-3x3
inception4_3x3conv1 = Conv2D(64, 1, name='inception4_3x3conv1')(inception3_concat) #9,9,64
inception4_3x3bn1 = BatchNormalization(axis=1, name='inception4_3x3bn1')(inception4_3x3conv1)
inception4_3x3relu1 = Activation('relu', name='inception4_3x3relu1')(inception4_3x3bn1)
inception4_3x3conv2 = Conv2D(96, 3, padding='SAME', name='inception4_3x3conv2')(inception4_3x3relu1) #9,9,96
inception4_3x3bn2 = BatchNormalization(axis=1, name='inception4_3x3bn2')(inception4_3x3conv2)
inception4_3x3relu2 = Activation('relu', name='inception4_3x3relu2')(inception4_3x3bn2)
inception4_3x3conv3 = Conv2D(96, 3, strides=(2, 2), name='inception4_3x3conv3')(inception4_3x3relu2) #4,4,96
inception4_3x3bn3 = BatchNormalization(axis=1, name='inception4_3x3bn3')(inception4_3x3conv3)
inception4_3x3relu3 = Activation('relu', name='inception4_3x3relu3')(inception4_3x3bn3)
# maxpool
# *** avg pool --> max pool
inception4_maxpool = MaxPooling2D((3, 3), strides=(2, 2), name='inception4_maxpool')(inception3_concat) #4,4,288
# *** no conv
# concat
inception4_concat = concatenate([inception4_3x3relu, inception4_3x3relu3, inception4_maxpool],
                                axis=-1, name='inception4_concat') #4,4,768

# Inception5
# 1x1
inception5_1x1conv = Conv2D(192, 1, name='inception5_1x1conv')(inception4_concat) #4,4,192
inception5_1x1bn = BatchNormalization(axis=1, name='inception5_1x1bn')(inception5_1x1conv)
inception5_1x1relu = Activation('relu', name='inception5_1x1relu')(inception5_1x1bn)
# 1x1-1x7-7x1
inception5_7x1conv1 = Conv2D(128, 1, name='inception5_7x1conv1')(inception4_concat) #4,4,128
inception5_7x1bn1 = BatchNormalization(axis=1, name='inception5_7x1bn1')(inception5_7x1conv1)
inception5_7x1relu1 = Activation('relu', name='inception5_7x1relu1')(inception5_7x1bn1)
inception5_7x1conv2 = Conv2D(128, (1, 7), padding='SAME', name='inception5_7x1conv2')(inception5_7x1relu1) #4,4,128
inception5_7x1bn2 = BatchNormalization(axis=1, name='inception5_7x1bn2')(inception5_7x1conv2)
inception5_7x1relu2 = Activation('relu', name='inception5_7x1relu2')(inception5_7x1bn2)
inception5_7x1conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception5_7x1conv3')(inception5_7x1relu2) #4,4,192
inception5_7x1bn3 = BatchNormalization(axis=1, name='inception5_7x1bn3')(inception5_7x1conv3)
inception5_7x1relu3 = Activation('relu', name='inception5_7x1relu3')(inception5_7x1bn3)
# 1x1-7x1-1x7-7x1-1x7
inception5_1x7conv1 = Conv2D(128, 1, name='inception5_1x7conv1')(inception4_concat) #4,4,128
inception5_1x7bn1 = BatchNormalization(axis=1, name='inception5_1x7bn1')(inception5_1x7conv1)
inception5_1x7relu1 = Activation('relu', name='inception5_1x7relu1')(inception5_1x7bn1)
inception5_1x7conv2 = Conv2D(128, (7, 1), padding='SAME', name='inception5_1x7conv2')(inception5_1x7relu1) #4,4,128
inception5_1x7bn2 = BatchNormalization(axis=1, name='inception5_1x7bn2')(inception5_1x7conv2)
inception5_1x7relu2 = Activation('relu', name='inception5_1x7relu2')(inception5_1x7bn2)
inception5_1x7conv3 = Conv2D(128, (1, 7), padding='SAME', name='inception5_1x7conv3')(inception5_1x7relu2) #4,4,128
inception5_1x7bn3 = BatchNormalization(axis=1, name='inception5_1x7bn3')(inception5_1x7conv3)
inception5_1x7relu3 = Activation('relu', name='inception5_1x7relu3')(inception5_1x7bn3)
inception5_1x7conv4 = Conv2D(128, (7, 1), padding='SAME', name='inception5_1x7conv4')(inception5_1x7relu3) #4,4,128
inception5_1x7bn4 = BatchNormalization(axis=1, name='inception5_1x7bn4')(inception5_1x7conv4)
inception5_1x7relu4 = Activation('relu', name='inception5_1x7relu4')(inception5_1x7bn4)
inception5_1x7conv5 = Conv2D(192, (1, 7), padding='SAME', name='inception5_1x7conv5')(inception5_1x7relu4) #4,4,192
inception5_1x7bn5 = BatchNormalization(axis=1, name='inception5_1x7bn5')(inception5_1x7conv5)
inception5_1x7relu5 = Activation('relu', name='inception5_1x7relu5')(inception5_1x7bn5)
# avgpool
inception5_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception5_avgpool')(inception4_concat) #4,4,768

inception5_avgpool_conv = Conv2D(192, 1, name='inception5_avgpool_conv')(inception5_avgpool) #4,4,192
inception5_avgpool_bn = BatchNormalization(axis=1, name='inception5_avgpool_bn')(inception5_avgpool_conv)
inception5_avgpool_relu = Activation('relu', name='inception5_avgpool_relu')(inception5_avgpool_bn)
# concat
inception5_concat = concatenate([inception5_1x1relu, inception5_7x1relu3, inception5_1x7relu5, inception5_avgpool_relu],
                                axis=-1, name='inception5_concat') #4,4,768
x = inception5_concat

# Inception6-7
for r in range(6, 8):
    i = str(r)
    # 1x1
    inception6_1x1conv = Conv2D(192, 1, name='inception'+i+'_1x1conv')(x) #4,4,192
    inception6_1x1bn = BatchNormalization(axis=1, name='inception'+i+'_1x1bn')(inception6_1x1conv)
    inception6_1x1relu = Activation('relu', name='inception'+i+'_1x1relu')(inception6_1x1bn)
    # 1x1-1x7-7x1
    inception6_7x1conv1 = Conv2D(128, 1, name='inception' + i + '_7x1conv1')(x) #4,4,128
    inception6_7x1bn1 = BatchNormalization(axis=1, name='inception' + i + '_7x1bn1')(inception6_7x1conv1)
    inception6_7x1relu1 = Activation('relu', name='inception' + i + '_7x1relu1')(inception6_7x1bn1)
    inception6_7x1conv2 = Conv2D(128, (1, 7), padding='SAME', name='inception' + i + '_7x1conv2')(inception6_7x1relu1) #4,4,128
    inception6_7x1bn2 = BatchNormalization(axis=1, name='inception' + i + '_7x1bn2')(inception6_7x1conv2)
    inception6_7x1relu2 = Activation('relu', name='inception' + i + '_7x1relu2')(inception6_7x1bn2)
    inception6_7x1conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception' + i + '_7x1conv3')(inception6_7x1relu2) #4,4,192
    inception6_7x1bn3 = BatchNormalization(axis=1, name='inception' + i + '_7x1bn3')(inception6_7x1conv3)
    inception6_7x1relu3 = Activation('relu', name='inception' + i + '_7x1relu3')(inception6_7x1bn3)
    # 1x1-7x1-1x7-7x1-1x7
    inception6_1x7conv1 = Conv2D(160, 1, name='inception' + i + '_1x7conv1')(x) #4,4,160
    inception6_1x7bn1 = BatchNormalization(axis=1, name='inception' + i + '_1x7bn1')(inception6_1x7conv1)
    inception6_1x7relu1 = Activation('relu', name='inception' + i + '_1x7relu1')(inception6_1x7bn1)
    inception6_1x7conv2 = Conv2D(160, (7, 1), padding='SAME', name='inception' + i + '_1x7conv2')(inception6_1x7relu1) #4,4,160
    inception6_1x7bn2 = BatchNormalization(axis=1, name='inception' + i + '_1x7bn2')(inception6_1x7conv2)
    inception6_1x7relu2 = Activation('relu', name='inception' + i + '_1x7relu2')(inception6_1x7bn2)
    inception6_1x7conv3 = Conv2D(160, (1, 7), padding='SAME', name='inception' + i + '_1x7conv3')(inception6_1x7relu2) #4,4,160
    inception6_1x7bn3 = BatchNormalization(axis=1, name='inception' + i + '_1x7bn3')(inception6_1x7conv3)
    inception6_1x7relu3 = Activation('relu', name='inception' + i + '_1x7relu3')(inception6_1x7bn3)
    inception6_1x7conv4 = Conv2D(160, (7, 1), padding='SAME', name='inception' + i + '_1x7conv4')(inception6_1x7relu3) #4,4,160
    inception6_1x7bn4 = BatchNormalization(axis=1, name='inception' + i + '_1x7bn4')(inception6_1x7conv4)
    inception6_1x7relu4 = Activation('relu', name='inception' + i + '_1x7relu4')(inception6_1x7bn4)
    inception6_1x7conv5 = Conv2D(192, (1, 7), padding='SAME', name='inception' + i + '_1x7conv5')(inception6_1x7relu4) #4,4,192
    inception6_1x7bn5 = BatchNormalization(axis=1, name='inception' + i + '_1x7bn5')(inception6_1x7conv5)
    inception6_1x7relu5 = Activation('relu', name='inception' + i + '_1x7relu5')(inception6_1x7bn5)
    # avgpool
    inception6_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception' + i + '_avgpool')(x) #4,4,768

    inception6_avgpool_conv = Conv2D(192, 1, name='inception' + i + '_avgpool_conv')(inception6_avgpool) #4,4,192
    inception6_avgpool_bn = BatchNormalization(axis=1, name='inception' + i + '_avgpool_bn')(inception6_avgpool_conv)
    inception6_avgpool_relu = Activation('relu', name='inception' + i + '_avgpool_relu')(inception6_avgpool_bn)

    # concat
    x = concatenate([inception6_1x1relu, inception6_7x1relu3, inception6_1x7relu5, inception6_avgpool_relu],
                    axis=-1, name='inception'+i+'_concat') #4,4,768
inception7_concat = x

# Inception8
# 1x1
inception8_1x1conv = Conv2D(192, 1, name='inception8_1x1conv')(inception7_concat) #4,4,192
inception8_1x1bn = BatchNormalization(axis=1, name='inception8_1x1bn')(inception8_1x1conv)
inception8_1x1relu = Activation('relu', name='inception8_1x1relu')(inception8_1x1bn)
# 1x1-1x7-7x1
inception8_7x1conv1 = Conv2D(192, 1, name='inception8_7x1conv1')(inception7_concat) #4,4,192
inception8_7x1bn1 = BatchNormalization(axis=1, name='inception8_7x1bn1')(inception8_7x1conv1)
inception8_7x1relu1 = Activation('relu', name='inception8_7x1relu1')(inception8_7x1bn1)
inception8_7x1conv2 = Conv2D(192, (1, 7), padding='SAME', name='inception8_7x1conv2')(inception8_7x1relu1) #4,4,192
inception8_7x1bn2 = BatchNormalization(axis=1, name='inception8_7x1bn2')(inception8_7x1conv2)
inception8_7x1relu2 = Activation('relu', name='inception8_7x1relu2')(inception8_7x1bn2)
inception8_7x1conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception8_7x1conv3')(inception8_7x1relu2) #4,4,192
inception8_7x1bn3 = BatchNormalization(axis=1, name='inception8_7x1bn3')(inception8_7x1conv3)
inception8_7x1relu3 = Activation('relu', name='inception8_7x1relu3')(inception8_7x1bn3)
# 1x1-7x1-1x7-7x1-1x7
inception8_1x7conv1 = Conv2D(192, 1, name='inception8_1x7conv1')(inception7_concat) #4,4,192
inception8_1x7bn1 = BatchNormalization(axis=1, name='inception8_1x7bn1')(inception8_1x7conv1)
inception8_1x7relu1 = Activation('relu', name='inception8_1x7relu1')(inception8_1x7bn1)
inception8_1x7conv2 = Conv2D(192, (7, 1), padding='SAME', name='inception8_1x7conv2')(inception8_1x7relu1) #4,4,192
inception8_1x7bn2 = BatchNormalization(axis=1, name='inception8_1x7bn2')(inception8_1x7conv2)
inception8_1x7relu2 = Activation('relu', name='inception8_1x7relu2')(inception8_1x7bn2)
inception8_1x7conv3 = Conv2D(192, (1, 7), padding='SAME', name='inception8_1x7conv3')(inception8_1x7relu2) #4,4,192
inception8_1x7bn3 = BatchNormalization(axis=1, name='inception8_1x7bn3')(inception8_1x7conv3)
inception8_1x7relu3 = Activation('relu', name='inception8_1x7relu3')(inception8_1x7bn3)
inception8_1x7conv4 = Conv2D(192, (7, 1), padding='SAME', name='inception8_1x7conv4')(inception8_1x7relu3) #4,4,192
inception8_1x7bn4 = BatchNormalization(axis=1, name='inception8_1x7bn4')(inception8_1x7conv4)
inception8_1x7relu4 = Activation('relu', name='inception8_1x7relu4')(inception8_1x7bn4)
inception8_1x7conv5 = Conv2D(192, (1, 7), padding='SAME', name='inception8_1x7conv5')(inception8_1x7relu4) #4,4,192
inception8_1x7bn5 = BatchNormalization(axis=1, name='inception8_1x7bn5')(inception8_1x7conv5)
inception8_1x7relu5 = Activation('relu', name='inception8_1x7relu5')(inception8_1x7bn5)
# avgpool
inception8_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception8_avgpool')(inception7_concat) #4,4,768

inception8_avgpool_conv = Conv2D(192, 1, name='inception8_avgpool_conv')(inception8_avgpool) #4,4,192
inception8_avgpool_bn = BatchNormalization(axis=1, name='inception8_avgpool_bn')(inception8_avgpool_conv)
inception8_avgpool_relu = Activation('relu', name='inception8_avgpool_relu')(inception8_avgpool_bn)
# concat
inception8_concat = concatenate([inception8_1x1relu, inception8_7x1relu3, inception8_1x7relu5, inception8_avgpool_relu],
                                axis=-1, name='inception8_concat') #4,4,768

# Inception9
# 1x1-3x3
inception9_3x3conv1 = Conv2D(192, 1, strides=(2, 2), name='inception9_3x3conv1')(inception8_concat) #2,2,192
inception9_3x3bn1 = BatchNormalization(axis=1, name='inception9_3x3bn1')(inception9_3x3conv1)
inception9_3x3relu1 = Activation('relu', name='inception9_3x3relu1')(inception9_3x3bn1)
inception9_3x3conv2 = Conv2D(320, 3, strides=(2, 2), padding='SAME', name='inception9_3x3conv2')(inception9_3x3relu1) #1,1,320
inception9_3x3bn2 = BatchNormalization(axis=1, name='inception9_3x3bn2')(inception9_3x3conv2)
inception9_3x3relu2 = Activation('relu', name='inception9_3x3relu2')(inception9_3x3bn2)
# 1x1-1x7-7x1-3x3
inception9_3x3_2_conv1 = Conv2D(192, 1, name='inception9_3x3_2_conv1')(inception8_concat) #4,4,192
inception9_3x3_2_bn1 = BatchNormalization(axis=1, name='inception9_3x3_2_bn1')(inception9_3x3_2_conv1)
inception9_3x3_2_relu1 = Activation('relu', name='inception9_3x3_2_relu1')(inception9_3x3_2_bn1)
inception9_3x3_2_conv2 = Conv2D(192, (1, 7), padding='SAME', name='inception9_3x3_2_conv2')(inception9_3x3_2_relu1) #4,4,192
inception9_3x3_2_bn2 = BatchNormalization(axis=1, name='inception9_3x3_2_bn2')(inception9_3x3_2_conv2)
inception9_3x3_2_relu2 = Activation('relu', name='inception9_3x3_2_relu2')(inception9_3x3_2_bn2)
inception9_3x3_2_conv3 = Conv2D(192, (7, 1), padding='SAME', name='inception9_3x3_2_conv3')(inception9_3x3_2_relu2) #4,4,192
inception9_3x3_2_bn3 = BatchNormalization(axis=1, name='inception9_3x3_2_bn3')(inception9_3x3_2_conv3)
inception9_3x3_2_relu3 = Activation('relu', name='inception9_3x3_2_relu3')(inception9_3x3_2_bn3)
inception9_3x3_2_conv4 = Conv2D(192, 3, strides=(2, 2), name='inception9_3x3_2_conv4')(inception9_3x3_2_relu3) #1,1,192
inception9_3x3_2_bn4 = BatchNormalization(axis=1, name='inception9_3x3_2_bn4')(inception9_3x3_2_conv4)
inception9_3x3_2_relu4 = Activation('relu', name='inception9_3x3_2_relu4')(inception9_3x3_2_bn4)
# maxpool
inception9_maxpool = MaxPooling2D((3, 3), strides=(2, 2), name='inception9_maxpool')(inception8_concat) #1,1,768
# concat
inception9_concat = concatenate([inception9_3x3relu2, inception9_3x3_2_relu4, inception9_maxpool],
                                axis=-1, name='inception9_concat') #1,1,1280
x = inception9_concat

# Inception10-11
for r in range(10, 12):
    i = str(r)
    # avgpool
    inception10_avgpool = AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='inception'+i+'_avgpool')(x) #1,1,1280

    inception10_avgpool_conv = Conv2D(192, 1, name='inception'+i+'_avgpool_conv')(inception10_avgpool) #1,1,192
    inception10_avgpool_bn = BatchNormalization(axis=1, name='inception'+i+'_avgpool_bn')(inception10_avgpool_conv)
    inception10_avgpool_relu = Activation('relu', name='inception'+i+'_avgpool_relu')(inception10_avgpool_bn)
    # 1x1
    inception10_1x1conv = Conv2D(320, 1, name='inception'+i+'_1x1conv')(x) #1,1,320
    inception10_1x1bn = BatchNormalization(axis=1, name='inception'+i+'_1x1bn')(inception10_1x1conv)
    inception10_1x1relu = Activation('relu', name='inception'+i+'_1x1relu')(inception10_1x1bn)
    # 1x1-1x3-3x1-concat
    inception10_3x1conv1 = Conv2D(384, 1, name='inception'+i+'_3x1conv1')(x) #1,1,384
    inception10_3x1bn1 = BatchNormalization(axis=1, name='inception'+i+'_3x1bn1')(inception10_3x1conv1)
    inception10_3x1relu1 = Activation('relu', name='inception'+i+'_3x1relu1')(inception10_3x1bn1)
    inception10_3x1conv2 = Conv2D(384, (1, 3), padding='SAME', name='inception'+i+'_3x1conv2')(inception10_3x1relu1) #1,1,384
    inception10_3x1bn2 = BatchNormalization(axis=1, name='inception'+i+'_3x1bn2')(inception10_3x1conv2)
    inception10_3x1relu2 = Activation('relu', name='inception'+i+'_3x1relu2')(inception10_3x1bn2)
    inception10_3x1conv3 = Conv2D(384, (3, 1), padding='SAME', name='inception'+i+'_3x1conv3')(inception10_3x1relu1) #1,1,384
    inception10_3x1bn3 = BatchNormalization(axis=1, name='inception'+i+'_3x1bn3')(inception10_3x1conv3)
    inception10_3x1relu3 = Activation('relu', name='inception'+i+'_3x1relu3')(inception10_3x1bn3)
    inception10_3x1concat = concatenate([inception10_3x1relu2, inception10_3x1relu3],
                                        axis=-1, name='inception'+i+'_3x1concat') #1,1,768
    # 1x1-3x3-1x3-3x1-concat
    inception10_3x1_2_conv1 = Conv2D(448, 1, name='inception'+i+'_3x1_2_conv1')(x) #1,1,448
    inception10_3x1_2_bn1 = BatchNormalization(axis=1, name='inception'+i+'_3x1_2_bn1')(inception10_3x1_2_conv1)
    inception10_3x1_2_relu1 = Activation('relu', name='inception'+i+'_3x1_2_relu1')(inception10_3x1_2_bn1)
    inception10_3x1_2_conv2 = Conv2D(384, 3, padding='SAME', name='inception'+i+'_3x1_2_conv2')(inception10_3x1_2_relu1) #1,1,384
    inception10_3x1_2_bn2 = BatchNormalization(axis=1, name='inception'+i+'_3x1_2_bn2')(inception10_3x1_2_conv2)
    inception10_3x1_2_relu2 = Activation('relu', name='inception'+i+'_3x1_2_relu2')(inception10_3x1_2_bn2)
    inception10_3x1_2_conv3 = Conv2D(384, (1, 3), padding='SAME', name='inception'+i+'_3x1_2_conv3')(inception10_3x1_2_relu2) #1,1,384
    inception10_3x1_2_bn3 = BatchNormalization(axis=1, name='inception'+i+'_3x1_2_bn3')(inception10_3x1_2_conv3)
    inception10_3x1_2_relu3 = Activation('relu', name='inception'+i+'_3x1_2_relu3')(inception10_3x1_2_bn3)
    inception10_3x1_2_conv4 = Conv2D(384, (3, 1), padding='SAME', name='inception'+i+'_3x1_2_conv4')(inception10_3x1_2_relu2) #1,1,384
    inception10_3x1_2_bn4 = BatchNormalization(axis=1, name='inception'+i+'_3x1_2_bn4')(inception10_3x1_2_conv4)
    inception10_3x1_2_relu4 = Activation('relu', name='inception'+i+'_3x1_2_relu4')(inception10_3x1_2_bn4)
    inception10_3x1_2_concat = concatenate([inception10_3x1_2_relu3, inception10_3x1_2_relu4],
                                           axis=-1, name='inception'+i+'_3x1_2_concat') #1,1,768
    # concat
    x = concatenate([inception10_avgpool_relu, inception10_1x1relu, inception10_3x1concat, inception10_3x1_2_concat],
                    axis=-1, name='inception'+i+'_concat') #1,1,2048
inception11_concat = x

# Fully connected (FC)
avgpool = GlobalAveragePooling2D(name='Global_avgpool')(inception11_concat) #2048
output_tensor = Dense(200, activation='softmax', name='output')(avgpool) #200

# Create a model
inceptionv3 = Model(input_tensor, output_tensor, name='inceptionv3')
inceptionv3.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
inceptionv3.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

# Train the model to adjust parameters to minimize the loss
inceptionv3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Test the model with test set
inceptionv3.evaluate(x_test, y_test, verbose=1)