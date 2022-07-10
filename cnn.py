# encoding: utf-8

# 导入TensorFlow和tf.keras
import tensorflow as tf
import numpy as np

import os, sys, shutil, time, json
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import data

MODEL_PATH = './models'

class DeCaptcha():
    def __init__(self):
        self.CreateModel()
    
    def CreateModel(self):
        self.x = x = tf.placeholder("float", shape=[None, 50, 90, 3])
        self.y_ = y_ = tf.placeholder("float", shape=[None, 5, 10+26+1])
        self.keep_prob = keep_prob = tf.placeholder("float") # dropout

        x_image = tf.reshape(x, [-1,50,90,3]) # reshape

        W_conv1 = weight_variable([3, 3, 3, 10]) # 第一层
        b_conv1 = bias_variable([10])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1) # 24*44*10

        W_conv2 = weight_variable([3, 3, 10, 20]) # 第二层
        b_conv2 = bias_variable([20])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2) # 11*21*20

        W_conv3 = weight_variable([4, 4, 20, 30]) # 第三层
        b_conv3 = bias_variable([30])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3) # 4*9*30

        W_fc1 = weight_variable([4*9*30, 256]) # 全连接1
        b_fc1 = bias_variable([256])
        h_pool2_flat = tf.reshape(h_pool3, [-1, 4*9*30])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([256, (10+26+1)*5]) # softmax
        b_fc2 = bias_variable([(10+26+1)*5])
        output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        shaped_output = tf.reshape(output, [-1, 5, 10+26+1])
        y_conv = tf.nn.softmax(shaped_output, axis=2)

        self.output = shaped_output

        self.cross_entropy = cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-10))
        self.train = train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(y_, 2))
        self.accuracy = accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.trueaccuracy = trueaccuracy = tf.reduce_mean(tf.cast(tf.reduce_all(correct_prediction, axis=1), "float"))

        self.sess = sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
    
    def Train(self):
        print('inited')
        train_data = data.train_data()

        BLOCK_SIZE = 32
        BLOCK_NUM = len(train_data['train_imgs'])//BLOCK_SIZE
        print(BLOCK_SIZE, BLOCK_NUM, len(train_data['train_imgs']))

        print('started')
        def onehot(data):
            assert len(data.shape) == 2
            d = np.zeros((data.shape[0], data.shape[1], 10+26+1), dtype=np.float32)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    d[i][j][data[i][j]] = 1
            return d
        try:
            saver = tf.train.Saver()
            best = 0
            for index in range(50000):
                bindex = index%BLOCK_NUM
                imgs = train_data['train_imgs'][bindex*BLOCK_SIZE: bindex*BLOCK_SIZE+BLOCK_SIZE]
                labels = train_data['train_labels'][bindex*BLOCK_SIZE: bindex*BLOCK_SIZE+BLOCK_SIZE]
                self.sess.run(self.train, feed_dict={self.x: imgs, self.y_: onehot(labels), self.keep_prob: 0.5})
                if index % 100 == 0:
                    loss = self.sess.run(self.cross_entropy, feed_dict={self.x: imgs, self.y_: onehot(labels), self.keep_prob: 1.0})
                    train_accuracy = self.sess.run(self.trueaccuracy, feed_dict={self.x: imgs, self.y_: onehot(labels), self.keep_prob: 1.0})
                    accuracy = self.sess.run(self.accuracy, feed_dict={self.x: train_data['test_imgs'], self.y_: onehot(train_data['test_labels']), self.keep_prob: 1.0})
                    trueaccuracy = self.sess.run(self.trueaccuracy, feed_dict={self.x: train_data['test_imgs'], self.y_: onehot(train_data['test_labels']), self.keep_prob: 1.0})
                    print('S', index, 'L', loss, 'tA', train_accuracy, 'A', accuracy, 'TA', trueaccuracy)
                    if best < trueaccuracy:
                        best = trueaccuracy
                        if os.path.exists(MODEL_PATH):
                            shutil.rmtree(MODEL_PATH)
                        os.mkdir(MODEL_PATH)
                        saver.save(self.sess, MODEL_PATH + '/x.model')
        except KeyboardInterrupt:
            pass
        print("best = %.4lf" % best)

    def Restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, MODEL_PATH + '/x.model')
        self.sess.graph.finalize()
    
    def Predict(self, filenames):
        if type(filenames) != list:
            filenames = [filenames]
        imgs = [data.loadData(filename) for filename in filenames]
        results = np.argmax(self.sess.run(self.output, feed_dict={self.x: imgs, self.keep_prob: 1.0}), -1)
        answers = []
        for line in results:
            s = ''
            for x in line:
                if 0 <= x and x < 10:
                    s += chr(x+ord('0'))
                elif 10 <= x and x < 10+26:
                    s += chr(x-10+ord('A'))
                else:
                    s += '_'
            answers.append(s)
        return answers

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

if __name__ == '__main__':
    c = DeCaptcha()
    if len(sys.argv) >= 2 and sys.argv[1] == 'train':
        c.Train()
    elif len(sys.argv) >= 3 and sys.argv[1] == 'predict':
        c.Restore()
        print(json.dumps(c.Predict(sys.argv[2:])))
    elif len(sys.argv) == 2 and sys.argv[1] == 'benchmark':
        c.Restore()
        N = 1000
        for t in [1, 10]:
            assert N%t == 0
            time_start = time.time()
            filenames = ['test.png'] * t
            for times in range(N//t):
                c.Predict(filenames)
            time_end = time.time()
            print('benchmark N = %d, t = %d, time cost: %.2f' % (N, t, time_end-time_start))
    else:
        print('usage : train / benchmark / predict filename1 filename2 ...')
