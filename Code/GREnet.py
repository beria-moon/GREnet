from __future__ import print_function
from pickle import LONG1
from re import A
from tkinter import Y

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import multiply, GlobalAveragePooling2D, AveragePooling2D, Cropping2D, Lambda, Dropout, Input, add, Conv2D, MaxPooling2D, UpSampling2D,concatenate, ZeroPadding2D, BatchNormalization, Activation,SpatialDropout2D
from tensorflow.keras.layers import ConvLSTM2D, Reshape,Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from keras.layers import Activation, Conv2D,Conv2DTranspose
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from tensorflow.keras.losses import binary_crossentropy
from keras.regularizers import l2


import tensorflow as tf
from torch import flatten
from loss_function import *
import os

from module import sspcab_layer

os.environ["CUDA_VISIBLE_DEVICES"]="0"
img_rows = 224
img_cols = 224                        

smooth = 1.
concat_axis = 3
bn_axis =3
def attention(inputs,stage):
    out = sspcab_layer(inputs,stage)
    return out    
def sk(inputs,stage):
    b,h,w,c = inputs.shape
    conv5_1 = Conv2D(256, (3, 3), padding="same", activation="relu", name=str(stage)+"conv5_1")(inputs)
    conv5_2 = Conv2D(256, (5, 5), padding="same", activation="relu", name=str(stage)+"conv5_2")(inputs)
    conv5_add = add([conv5_1, conv5_2])
    gp = AveragePooling2D(pool_size=(h,w), name=str(stage)+"ap")(conv5_add)
    weight = Conv2D(256, kernel_size=(1, 1), strides=(1,1), padding="valid",name=str(stage)+"weight_s")(gp)
    weight = BatchNormalization(axis=concat_axis)(weight)
    weight = Activation("relu")(weight)
    weight_2 = Conv2D(256*2, kernel_size=(1, 1), strides=(1,1), padding="valid",name=str(stage)+"weight_z")(weight)
    weight_2 = Reshape([1,1,256,2])(weight_2)
    weight_2 = Activation("softmax", name=str(stage)+"weight_z_softmax")(weight_2)
    conv5_1 = Reshape([h,w,256,1])(conv5_1)
    conv5_2 = Reshape([h,w,256,1])(conv5_2)
    conv5_concat = concatenate([conv5_1, conv5_2],axis=4, name=str(stage)+"conv5_concat")
    conv5_mul = multiply([conv5_concat, weight_2],name=str(stage)+"conv5_mul")
    conv5_last = Lambda(lambda x: tf.reduce_sum(x, axis=-1, name=str(stage)+'sum'))(conv5_mul)
    return conv5_last
   
def unet(inputs, num):
    #inputs = Input((img_rows, img_cols, 3))
    conv1= Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    #conv1 = SpatialDropout2D(dropout_rate)(conv1)
    conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    #conv2 = SpatialDropout2D(dropout_rate)(conv2)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
    #conv3 = SpatialDropout2D(dropout_rate)(conv3)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    #conv4 = SpatialDropout2D(dropout_rate)(conv4)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
    #conv5 = SpatialDropout2D(dropout_rate)(conv5)
    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv5) 
    # conv5_last = sk(conv5, num)
   
    # conv_att = Conv2D(256,(1,1),activation="relu",padding="valid")(conv5_last)
    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up_conv5,conv4], axis=concat_axis)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
    #conv6 = SpatialDropout2D(dropout_rate)(conv6)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)
    

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
    #conv7 = SpatialDropout2D(dropout_rate)(conv7)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)
    
    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(up8)
    #conv8 = SpatialDropout2D(dropout_rate)(conv8)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(up9)
    #conv9 = SpatialDropout2D(dropout_rate)(conv9)
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    return conv10,conv5

def unet_lstm(inputs, feature, num):
    b,h,w,c = inputs.shape
    #inputs = Input((img_rows, img_cols, 3))
    conv1= Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    #conv1 = SpatialDropout2D(dropout_rate)(conv1)
    conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    #conv2 = SpatialDropout2D(dropout_rate)(conv2)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
    #conv3 = SpatialDropout2D(dropout_rate)(conv3)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    #conv4 = SpatialDropout2D(dropout_rate)(conv4)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
    reshape_1 = Reshape(target_shape=(1, h//16, w//16 ,256))(feature)
    reshape_2 = Reshape(target_shape=(1, h//16, w//16, 256))(conv5)
    concat_lstm = concatenate([reshape_1, reshape_2], axis=1)
    conv5_lstm = ConvLSTM2D(256, 3, activation="relu", padding="same", return_sequences=False)(concat_lstm)
    conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv5_lstm)# convlstm
    conv5 = concatenate([feature,conv5], axis = concat_axis)
    conv5 = Conv2D(256,(1,1),activation='relu',padding='same')(conv5)
    conv5_last = sk(conv5, num)
   
    # conv_att = Conv2D(256,(1,1),activation="relu",padding="valid")(conv5_last)
    # up_conv5 = UpSampling2D(size=(2, 2))(conv_att)
    up_conv5 = UpSampling2D(size=(2, 2))(conv5_last)
    up6 = concatenate([up_conv5,conv4], axis=concat_axis)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
    #conv6 = SpatialDropout2D(dropout_rate)(conv6)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)
    

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
    #conv7 = SpatialDropout2D(dropout_rate)(conv7)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)
    
    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(up8)
    #conv8 = SpatialDropout2D(dropout_rate)(conv8)
    conv8 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(up9)
    #conv9 = SpatialDropout2D(dropout_rate)(conv9)
    conv9 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    return conv10, conv5_last

act = "relu"
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):#activation=act,, dilation_rate=1

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act,name='conv'+stage+'_1',
               kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    # x = SpatialDropout2D(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act,name='conv'+stage+'_2', 
               kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    #x = SpatialDropout2D(dropout_rate, name='dp'+stage+'_2')(x)

    return x

def Nest_Net(inputs,num):

    nb_filter = [16,32,64,128,256]#
    conv1_1 = standard_unit(inputs, stage=str(num)+'11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage=str(num)+'21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name=str(num)+'pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name=str(num)+'up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name=str(num)+'merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage=str(num)+'12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage=str(num)+'31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name=str(num)+'pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name=str(num)+'up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name=str(num)+'merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage=str(num)+'22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name=str(num)+'up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name=str(num)+'merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage=str(num)+'13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage=str(num)+'41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name=str(num)+'pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name=str(num)+'up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name=str(num)+'merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage=str(num)+'32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name=str(num)+'up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name=str(num)+'merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage=str(num)+'23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name=str(num)+'up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name=str(num)+'merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage=str(num)+'14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage=str(num)+'51', nb_filter=nb_filter[4])
    conv5_last = sk(conv5_1, num)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name=str(num)+'up42', padding='same')(conv5_last)
    conv4_2 = concatenate([up4_2, conv4_1], name=str(num)+'merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage=str(num)+'42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name=str(num)+'up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name=str(num)+'merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage=str(num)+'33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name=str(num)+'up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name=str(num)+'merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage=str(num)+'24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name=str(num)+'up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name=str(num)+'merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage=str(num)+'15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(1, (1, 1), activation='sigmoid', name=str(num)+'output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(1, (1, 1), activation='sigmoid', name=str(num)+'output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(1, (1, 1), activation='sigmoid', name=str(num)+'output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', name=str(num)+'output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    return nestnet_output_4,conv5_last      


    
def get_unet():
    inputs_1 = Input((img_rows, img_cols, 3))
 
    output_1,feature_1 = unet(inputs_1,1)
    output_2, feature_2 = unet_lstm(inputs_1, feature_1, 2)
    output_3, feature_3 = unet_lstm(inputs_1, feature_2, 3)


 
    

    model = Model(inputs=inputs_1, outputs=[output_1,output_2,output_3])
  
    model.compile(optimizer=Adam(learning_rate=0.0001), loss={'conv2d_18':jaccard_l1_bce,'conv2d_37':jaccard_l1_bce,"conv2d_56":jaccard_l1_bce} ,
                  loss_weights={'conv2d_18':1,'conv2d_37':1})
    return model



def train_and_predict():
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    imgs_train = np.load("")
    imgs_train = np.transpose(imgs_train, (0, 2, 3, 1))
    imgs_train = np.array(imgs_train,dtype=np.float32)
    imgs_mask_train=np.load("")
    imgs_mask_train = np.transpose(imgs_mask_train, (0, 2, 3, 1))
    imgs_mask_train = np.array(imgs_mask_train,dtype=np.uint8)
    print("the dtype of mask:", imgs_mask_train.dtype,imgs_mask_train.shape)
   
 
    
    
    output_1_multiple=np.load("")
    output_1_multiple = np.transpose(output_1_multiple, (0, 2, 3, 1))
    output_1_multiple = np.array(output_1_multiple,dtype=np.uint8)
    #print("the dtype of mask:", output_1_multiple.dtype)
    #output_1_multiple = np.reshape(output_1_multiple, (20000,1,160,224,1))
    output_11_multiple=np.load("")
    # output_11_multiple = np.transpose(output_11_multiple, (0, 2, 3, 1))
    output_11_multiple = np.array(output_11_multiple,dtype=np.float32)
    print("the dtype of mask:", output_11_multiple.dtype)
    #output_1_multiple = np.reshape(output_1_multiple, (20000,1,160,224,1))
    output_2_multiple=np.load("")
    output_2_multiple = np.transpose(output_2_multiple, (0, 2, 3, 1))
    output_2_multiple = np.array(output_2_multiple,dtype=np.uint8)
    print("the dtype of mask:", output_2_multiple.dtype)
    
    output_3_multiple=np.load("")
    output_3_multiple = np.transpose(output_3_multiple, (0, 2, 3, 1))
    output_3_multiple = np.array(output_3_multiple,dtype=np.uint8)
    print("the dtype of mask:", output_3_multiple.dtype)
    
    output_4_multiple=np.load("")
    output_4_multiple = np.transpose(output_4_multiple, (0, 2, 3, 1))
    output_4_multiple = np.array(output_4_multiple,dtype=np.uint8)
    print("the dtype of mask:", output_4_multiple.dtype)
    # output_4_multiple = np.reshape(output_4_multiple, (21500,1,160,224,1))
    output_44_multiple=np.load("")
    # output_44_multiple = np.transpose(output_44_multiple)
    output_44_multiple = np.array(output_44_multiple,dtype=np.float32)
    print("the dtype of mask:", output_44_multiple.dtype)
    
    
    
    
    
   

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    
  

    #model.load_weights("/home/xingyanyan/unet_lstm/skin_lesion_2017/weights/multiple_2-26.hdf5")
    model.summary()
    
    model_checkpoint =ModelCheckpoint('/multiple_2-{epoch:02d}.hdf5',period=1, save_weights_only=False)
    # train1 = [output_1_multiple,imgs_mask_train,  output_2_multiple, imgs_mask_train,output_4_multiple, imgs_mask_train]
    # train2 =[ imgs_mask_train,imgs_mask_train]
    train4 = [output_1_multiple,imgs_mask_train, output_2_multiple, imgs_mask_train, output_3_multiple, imgs_mask_train,output_4_multiple, imgs_mask_train]

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    print('-'*30)
    hist=model.fit(imgs_train,train4, batch_size=16, epochs=40,
                   verbose=1, shuffle=True,callbacks=[model_checkpoint],initial_epoch=0)
    with open('/train_ds.txt','w')as f:
        f.write(str(hist.history))

if __name__ == '__main__':
    train_and_predict()
    
