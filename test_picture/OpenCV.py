import os
import sys
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
def read_images(path, sz = None):
    c = 0
    x, y = [], []
    names=[]
    for dirname, dirnames, filenames in os.walk(path):
        names.extend(dirnames)
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if not filename.endswith('.pgm'):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)#input 200*200 picture
                    if sz is not None:
                        im = cv2.resize(im,(200,200))               
                    x.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except:
                    print("Unexpected error:",sys.exc_info()[0])
            c = c + 1
    print('---data label information---')
    x=np.asarray(x)
    y=np.asarray(y)
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)  
    print(names)
    print('-'*30)
    return [x, y],names
x=read_images(r'C:\Users\max21\Desktop\Python\OpenCV\p77_save_data')[0][0]
y=read_images(r'C:\Users\max21\Desktop\Python\OpenCV\p77_save_data')[0][1]
#shuffle the images and labels
num_example=x.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
x=x[arr]
y=y[arr]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#tensorflow
w=128
h=128
c=3
X=tf.placeholder(tf.float32,shape=[None,w,h,c],name='X')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def CNNlayer():
    #第一个卷积层（128——>64)
    conv1=tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
 
    #第二个卷积层(64->32)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
 
    #第三个卷积层(32->16)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
 
    #第四个卷积层(16->8)
    conv4=tf.layers.conv2d(
          inputs=pool3,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
 
    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])
 
    #全连接层
    dense1 = tf.layers.dense(inputs=re1, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits= tf.layers.dense(inputs=dense2, 
                            units=3,
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits


logits=CNNlayer()
loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




