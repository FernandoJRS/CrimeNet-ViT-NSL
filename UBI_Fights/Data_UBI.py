import cv2 as cv2
import json
import numpy as np
import tensorflow as tf
import random
import os
import csv
from ViT import *

path_base = 'E:/UBI-Fights/annotation/'
path_videos = 'E:/UBI-Fights/videos/'

annotations = os.listdir(path_base)
videos = os.listdir(path_videos + 'fight/') + os.listdir(path_videos + 'normal/')
test_videos = [l[0] + '.mp4' for l in list(csv.reader(open('E:/UBI-Fights/test_videos.csv')))]
train_videos = [p for p in videos if [p] not in test_videos]

width = 224
height = 224
channels = 3
num_classes = 2

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


train_total_op, train_total_rgb = [], []
for i in range(len(train_videos)):
    if 'F' in train_videos[i]:
        print('Loading: ' + path_videos + 'fight/' + train_videos[i])
        video_frames_op = read_video_optical_flow(path_videos + 'fight/' + train_videos[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos + 'fight/' + train_videos[i], 20, 20, resize=True)
    else:
        print('Loading: ' + path_videos + 'normal/' + train_videos[i])
        video_frames_op = read_video_optical_flow(path_videos + 'normal/' + train_videos[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos + 'normal/' + train_videos[i], 20, 20, resize=True)
        
    frames_label = list(csv.reader(open(path_base + train_videos[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames_op)):
        fr_op = video_frames_op[j]
        fr_rgb = video_frames_rgb[j]
        label = frames_label[j][0]
        train_total_op.append((fr_op, label))
        train_total_rgb.append((fr_rgb, label))

temp = list(zip(train_total_op, train_total_rgb))
random.shuffle(temp)
train_total_op, train_total_rgb = zip(*temp)

test_total_op, test_total_rgb = [], []
for i in range(len(test_videos)):
    if 'F' in test_videos[i]:
        print('Loading: ' + path_videos + 'fight/' + test_videos[i])
        video_frames_op = read_video_optical_flow(path_videos + 'fight/' + test_videos[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos + 'fight/' + test_videos[i], 20, 20, resize=True)
    else:
        print('Loading: ' + path_videos + 'normal/' + test_videos[i])
        video_frames_op = read_video_optical_flow(path_videos + 'normal/' + test_videos[i], 20, 20, resize=True)
        video_frames_rgb = read_video(path_videos + 'normal/' + test_videos[i], 20, 20, resize=True)
        
    frames_label = list(csv.reader(open(path_base + test_videos[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames_op)):
        fr_op = video_frames_op[j]
        fr_rgb = video_frames_rgb[j]
        label = frames_label[j][0]
        test_total_op.append((fr_op, label))
        test_total_rgb.append((fr_rgb, label))

temp = list(zip(test_total_op, test_total_rgb))
random.shuffle(temp)
test_total_op, test_total_rgb = zip(*temp)


def generatorData(dataset_op, dataset_rgb, classes, batch_size=16):
    while True:
        for count in range(int(len(dataset_op) / batch_size)):
            batch_start = batch_size * count
            batch_stop = batch_size + (batch_size * count)
            lx_op = []
            lx_rgb = []
            ly = []

            for i in range(batch_start, batch_stop):
                frame_op = cv2.resize(dataset_op[i][0], (width, height))
                frame_op = (frame_op.astype('float32') - 127.5) / 127.5

                frame_rgb = cv2.resize(dataset_rgb[i][0], (width, height))
                frame_rgb = (frame_rgb.astype('float32') - 127.5) / 127.5

                label = dataset_op[i][1]

                lx_op.append(frame_op)
                lx_rgb.append(frame_rgb)
                ly.append(label)

            x_op = np.array(lx_op).astype('float32')
            x_rgb = np.array(lx_rgb).astype('float32')

            y = np.array(ly).astype('float32')
            y = tf.keras.utils.to_categorical(y, num_classes=classes, dtype='float32')

            x_op = tf.convert_to_tensor(x_op)
            x_rgb = tf.convert_to_tensor(x_rgb)
            y = tf.convert_to_tensor(y)

            yield {'feature_1': x_op, 'feature_2': x_rgb, 'label': y}
