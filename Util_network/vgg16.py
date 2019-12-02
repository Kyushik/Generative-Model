import inspect
import os

import numpy as np
import tensorflow as tf
import time
import glob
import matplotlib.pyplot as plt
import cv2

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path,allow_pickle=True, encoding='latin1').item()
        print("npy file loaded")

    def build(self, bgr):

        start_time = time.time()
        print("build model started")
        bgr_scaled = (bgr+1.0)/2.0 * 255.0

        with tf.variable_scope('vgg_net',reuse=tf.AUTO_REUSE):

            self.conv1_1 = self.conv_layer(bgr_scaled, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")



def per_loss(img1,img2):

    loss=tf.square(img1-img2)

    img_shape=tf.shape(loss)

    img_shape=tf.cast(img_shape,tf.float32)

    loss_norm=tf.reduce_mean(loss/(img_shape[1]*img_shape[2]*img_shape[3]))

    return loss_norm
