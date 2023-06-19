# coding=utf8
#from models import c3d_model
from sre_constants import CATEGORY
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime
from glob import glob
from tqdm import tqdm
import time

from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
from KeypointsDetector import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def c3d_model():
    input_shape = (112,112,16,3)
    weight_decay = 0.005
    nb_classes = 20

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model


def M1_test(path):
    now = datetime.now()
    times = str(now)
    test_date=str(datetime.today().month) +'.'+ str(datetime.today().day)
    print(path)
    video_name=path.split('/')[-1]


    fm=open('./input_data/index.txt', 'r')
    main_names = fm.readlines()
    CATEGORY = video_name[0:4]
    if not os.path.exists('./test_log'):
        os.mkdir('./test_log')
    if not os.path.exists('./test_log/'+CATEGORY):
        os.mkdir('./test_log/'+CATEGORY)

    file_name = video_name.split('.')[0]
    fw =open('./test_log/'+CATEGORY+'/'+file_name+'_M1_'+times[0:19]+'_.txt', 'w')
    # init model
    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model = tf.keras.models.load_model('./input_data/epoch10_temp_weights_c3d.h5')

    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(3))
    height = int(cap.get(4))

    clip = []
    main_count_list = [0 for i in range(len(main_names))]
    scene=0
    start = time.time()
    keypoints = []
    for i in tqdm(range(fps)):
        ret, frame = cap.read()
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))


                pred_main = model.predict(inputs)
                main_label = np.argmax(pred_main[0])
                main_count_list[main_label]=main_count_list[main_label]+1
                fw.write(main_names[main_label].split(' ')[1].strip()+" prob: %.4f" % pred_main[0][main_label]+'\n')
                clip.pop(0)

            # detect keypoints by mp
            kps = keypoint_detector(tmp)
            if len(kps) != 0:
                keypoints.append(keypoint_detector(tmp))
            else:
                keypoints.append([])


    end = datetime.now()
    end_time = str(end)
    ftw = open('./test_log/'+CATEGORY+'/'+file_name+'_M1_'+end_time[0:19]+'_total.txt', 'w')
    ftw.write(video_name+'\n')
    ftw.write('영상 '+str(fps-15)+' 프레임 중 ')
    main_mode_label = np.argmax(main_count_list)
    ftw.write(main_names[main_mode_label].split(' ')[-1].strip()+" 검출 "+str(main_count_list[main_mode_label])+" 프레임 ")\

    main_frame_prod=main_count_list[main_mode_label]/(fps-15)*100

    return_value =main_names[main_mode_label].split(' ')[-1].strip()
    ftw.write(str(int(main_frame_prod))+'%\n')
    for corr_main_label in range(len(main_names)):
        if video_name==main_names[corr_main_label].split(' ')[-1].strip()!=main_names[main_mode_label].split(' ')[-1].strip():
            main_frame_prod=main_count_list[corr_main_label]/(fps-15)*100
            ftw.write('\t\t\t\t\t\t\t'+main_names[corr_main_label].split(' ')[-1].strip()+" 검출 "+str(main_count_list[corr_main_label])+" 프레임 "+str(int(main_frame_prod))+'%\n')
    return (return_value, keypoints)



