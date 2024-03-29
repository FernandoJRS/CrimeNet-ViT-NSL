import cv2 as cv2
import numpy as np
import random
import tensorflow as tf
import os
from sklearn.model_selection import StratifiedShuffleSplit
from ViT import *

path_videos_nv = '/home/user/work/data/RLVS/NonViolence/'
path_videos_v = '/home/user/work/data/RLVS/Violence/'

videos_v = os.listdir(path_videos_v)
videos_nv = os.listdir(path_videos_nv)

label_videos_v = [1 for i in videos_v]
label_videos_nv = [0 for j in videos_nv]

videos = videos_v + videos_nv
label_videos = label_videos_v + label_videos_nv

width = 224
height = 224
channels = 3


def read_video_optical_flow(vid, width, height, resize=False):
    video_frames_optical_flow = list()
    i = 0
    cap = cv2.VideoCapture(vid)
    ret1, frame1 = cap.read()
    if resize:
        frame1 = cv2.resize(frame1, (width, height))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret2, frame2 = cap.read()
        if ret2:
            if resize:
                frame2 = cv2.resize(frame2, (width, height))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bgr = np.reshape(bgr, (width, height, channels))
            video_frames_optical_flow.append(bgr)
        else:
            break
        i += 1
        prvs = next
    cap.release()
    cv2.destroyAllWindows()
    return video_frames_optical_flow


def read_video(vid, width, height, resize=False):
    video_frames = list()
    cap = cv2.VideoCapture(vid)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if resize:
                frame = cv2.resize(frame, (width, height))
                frame = np.reshape(frame, (width, height, channels))
            video_frames.append(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return video_frames


X_train = []
y_train = []
X_valid_test = []
y_valid_test = []
X_test = []
y_test = []
X_valid = []
y_valid = []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_valid_index in split.split(videos, label_videos):
    for ti in train_index:
        X_train.append(videos[ti])
        y_train.append(label_videos[ti])

    for tsi in test_valid_index:
        X_valid_test.append(videos[tsi])
        y_valid_test.append(label_videos[tsi])

split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for test_index, valid_index in split2.split(X_valid_test, y_valid_test):
    for tssi in test_index:
        X_test.append(X_valid_test[tssi])
        y_test.append(y_valid_test[tssi])

    for tvi in valid_index:
        X_valid.append(X_valid_test[tvi])
        y_valid.append(y_valid_test[tvi])

train_total_op = []
train_total_rgb = []
for i in range(len(X_train)):
    if 'NV' in X_train[i]:
        print('Loading Training: ' + path_videos_nv + X_train[i])
        video_frames_op = read_video_optical_flow(path_videos_nv + X_train[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos_nv + X_train[i], 20, 20, resize=True)
    else:
        print('Loading Training: ' + path_videos_v + X_train[i])
        video_frames_op = read_video_optical_flow(path_videos_v + X_train[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos_v + X_train[i], 20, 20, resize=True)
    for j in range(len(video_frames_op)):
        fr_op = video_frames_op[j]
        fr_rgb = video_frames_rgb[j]
        if 'NV' in X_train:
            train_total_op.append((fr_op, 0))
            train_total_rgb.append((fr_rgb, 0))
        else:
            train_total_op.append((fr_op, 1))
            train_total_rgb.append((fr_rgb, 1))

validation_total_op = []
validation_total_rgb = []
for i in range(len(X_valid)):
    if 'NV' in X_valid[i]:
        print('Loading Validation: ' + path_videos_nv + X_valid[i])
        video_frames_op = read_video_optical_flow(path_videos_nv + X_valid[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos_nv + X_valid[i], 20, 20, resize=True)
    else:
        print('Loading Validation: ' + path_videos_v + X_valid[i])
        video_frames_op = read_video_optical_flow(path_videos_v + X_valid[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos_v + X_valid[i], 20, 20, resize=True)
    for j in range(len(video_frames_op)):
        fr_op = video_frames_op[j]
        fr_rgb = video_frames_rgb[j]
        if 'NV' in X_valid:
            validation_total_op.append((fr_op, 0))
            validation_total_rgb.append((fr_rgb, 0))
        else:
            validation_total_op.append((fr_op, 1))
            validation_total_rgb.append((fr_rgb, 1))

test_total_op = []
test_total_rgb = []
for i in range(len(X_test)):
    if 'NV' in X_test[i]:
        print('Loading Test: ' + path_videos_nv + X_test[i])
        video_frames_op = read_video_optical_flow(path_videos_nv + X_test[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos_nv + X_test[i], 20, 20, resize=True)
    else:
        print('Loading Test: ' + path_videos_nv + X_test[i])
        video_frames_op = read_video_optical_flow(path_videos_v + X_test[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos_v + X_test[i], 20, 20, resize=True)
    for j in range(len(video_frames_op)):
        fr_op = video_frames_op[j]
        fr_rgb = video_frames_rgb[j]
        if 'NV' in X_test:
            test_total_op.append((fr_op, 0))
            test_total_rgb.append((fr_rgb, 0))
        else:
            test_total_op.append((fr_op, 1))
            test_total_rgb.append((fr_rgb, 1))
            
temp = list(zip(train_total_op, train_total_rgb))
random.shuffle(temp)
train_total_op, train_total_rgb = zip(*temp)

# Train

def generatorTrainData(batch_size_train=16):
    while True:
        for count in range(int(len(train_total_op) / batch_size_train)):
            batch_start = batch_size_train * count
            batch_stop = batch_size_train + (batch_size_train * count)
            lx_op = []
            lx_rgb = []
            ly = []

            for i in range(batch_start, batch_stop):
                frame_op = cv2.resize(train_total_op[i][0], (width, height))
                frame_op = (frame_op.astype('float32') - 127.5) / 127.5

                frame_rgb = cv2.resize(train_total_rgb[i][0], (width, height))
                frame_rgb = (frame_rgb.astype('float32') - 127.5) / 127.5

                label = train_total_op[i][1]

                lx_op.append(frame_op)
                lx_rgb.append(frame_rgb)
                ly.append(label)

            x_op = np.array(lx_op).astype('float32')
            x_rgb = np.array(lx_rgb).astype('float32')

            y = np.array(ly).astype('float32')
            y = tf.keras.utils.to_categorical(y, num_classes=num_classes, dtype='float32')

            x_op = tf.convert_to_tensor(x_op)
            x_rgb = tf.convert_to_tensor(x_rgb)
            y = tf.convert_to_tensor(y)

            yield {'feature_1': x_op, 'feature_2': x_rgb, 'label': y}


# Validation

def generatorValidationData(batch_size_train=16):
    while True:
        for count in range(int(len(validation_total_op) / batch_size_train)):
            batch_start = batch_size_train * count
            batch_stop = batch_size_train + (batch_size_train * count)
            lx_op = []
            lx_rgb = []
            ly = []

            for i in range(batch_start, batch_stop):
                frame_op = cv2.resize(validation_total_op[i][0], (width, height))
                frame_op = (frame_op.astype('float32') - 127.5) / 127.5

                frame_rgb = cv2.resize(validation_total_rgb[i][0], (width, height))
                frame_rgb = (frame_rgb.astype('float32') - 127.5) / 127.5

                label = validation_total_op[i][1]

                lx_op.append(frame_op)
                lx_rgb.append(frame_rgb)
                ly.append(label)

            x_op = np.array(lx_op).astype('float32')
            x_rgb = np.array(lx_rgb).astype('float32')

            y = np.array(ly).astype('float32')
            y = tf.keras.utils.to_categorical(y, num_classes=num_classes, dtype='float32')

            x_op = tf.convert_to_tensor(x_op)
            x_rgb = tf.convert_to_tensor(x_rgb)
            y = tf.convert_to_tensor(y)

            yield {'feature_1': x_op, 'feature_2': x_rgb, 'label': y}


# Test

def generatorTestData(batch_size_test=16):
    while True:
        for count in range(int(len(test_total_op) / batch_size_test)):
            batch_start = batch_size_test * count
            batch_stop = batch_size_test + (batch_size_test * count)
            lx_op = []
            lx_rgb = []
            ly = []

            for i in range(batch_start, batch_stop):
                frame_op = cv2.resize(test_total_op[i][0], (width, height))
                frame_op = (frame_op.astype('float32') - 127.5) / 127.5

                frame_rgb = cv2.resize(test_total_rgb[i][0], (width, height))
                frame_rgb = (frame_rgb.astype('float32') - 127.5) / 127.5

                label = test_total_op[i][1]

                lx_op.append(frame_op)
                lx_rgb.append(frame_rgb)
                ly.append(label)

            x_op = np.array(lx_op).astype('float32')
            x_rgb = np.array(lx_rgb).astype('float32')

            y = np.array(ly).astype('float32')
            y = tf.keras.utils.to_categorical(y, num_classes=num_classes, dtype='float32')

            x_op = tf.convert_to_tensor(x_op)
            x_rgb = tf.convert_to_tensor(x_rgb)
            y = tf.convert_to_tensor(y)

            yield {'feature_1': x_op, 'feature_2': x_rgb, 'label': y}
