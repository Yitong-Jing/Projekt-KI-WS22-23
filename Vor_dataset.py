import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

X1 = tf.compat.v1.placeholder(tf.uint8)
Y1 = tf.compat.v1.placeholder(tf.int32)
X2 = tf.compat.v1.placeholder(tf.uint8)
Y2 = tf.compat.v1.placeholder(tf.int32)

A_train = tf.cast(tf.pad(X1, [[0,0],[0,0],[16,16],[0,0]]),dtype=tf.float32)
B_train = tf.cast(tf.one_hot(Y1, 4),dtype=tf.float32)
A_test = tf.cast(tf.pad(X2, [[0,0],[0,0],[16,16],[0,0]]),dtype=tf.float32)
B_test = tf.cast(tf.one_hot(Y2, 4),dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess :
    init.run()
    A_tn = A_train.eval(feed_dict={X1:X_train})
    B_tn = B_train.eval(feed_dict={Y1: Y_train})
    A_ts = A_test.eval(feed_dict={X2: X_test})
    B_ts = B_test.eval(feed_dict={Y2: Y_test})

np.save("X_train11.npy",A_tn)
np.save("Y_train11.npy",B_tn)
np.save("X_test11.npy",A_ts)
np.save("Y_test11.npy",B_ts)

print(A_tn.shape)
print(type(A_tn))
print(A_tn.dtype)


print(A_ts.shape)
print(type(A_ts))
print(A_ts.dtype)

print(B_tn.shape)
print(type(B_tn))
print(B_tn.dtype)

print(B_ts.shape)
print(type(B_ts))
print(B_ts.dtype)
