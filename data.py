# encoding: utf-8

# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras

import os, sys, random, re
import numpy as np
import cv2

def loadImage(filename):
    im = cv2.imread(filename, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (90, 50))
    return im

def loadData(filename):
    return loadImage(filename) / 256.0

def train_data():
    if os.path.exists('./dataed/train_imgs.npy'):
        train_imgs = np.load('./dataed/train_imgs.npy')
        train_labels = np.load('./dataed/train_labels.npy')
        test_imgs = np.load('./dataed/test_imgs.npy')
        test_labels = np.load('./dataed/test_labels.npy')
    else:
        PATHS = [
            os.path.join('images')
        ]
        FILES = []
        for p in PATHS:
            FILES.extend([os.path.join(p, x) for x in os.listdir(p)])
        random.shuffle(FILES)

        data = np.zeros((len(FILES), 50, 90, 3), dtype=np.float32)
        labels = np.zeros((len(FILES), 5), dtype=np.int32)
        for i in range(len(FILES)):
            data[i] = loadData(FILES[i])
            name = re.compile(r'(.*?)\.').search(os.path.basename(FILES[i])).group(1)
            for id in range(5):
                if id < len(name):
                    label = name[id]
                else:
                    label = '_'
                if '0' <= label and label <= '9':
                    labels[i][id] = ord(label) - ord('0')
                elif 'A' <= label and label <= 'Z':
                    labels[i][id] = ord(label) - ord('A') + 10
                else:
                    labels[i][id] = 26 + 10

        TRAIN_SIZE = int(round(len(FILES)*0.8))
        train_imgs = data[:TRAIN_SIZE]
        train_labels = labels[:TRAIN_SIZE]
        test_imgs = data[TRAIN_SIZE:]
        test_labels = labels[TRAIN_SIZE:]

        if not os.path.exists('dataed'): os.makedirs('dataed')
        np.save('./dataed/train_imgs.npy', train_imgs)
        np.save('./dataed/train_labels.npy', train_labels)
        np.save('./dataed/test_imgs.npy', test_imgs)
        np.save('./dataed/test_labels.npy', test_labels)

    return {
        'train_imgs': train_imgs,
        'train_labels': train_labels,
        'test_imgs': test_imgs,
        'test_labels': test_labels
    }

if __name__ == '__main__':
    train_data()
