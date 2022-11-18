import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

X_train = np.load("X_train11.npy")
Y_train = np.load("Y_train11.npy")
X_test = np.load("X_test11.npy")
Y_test = np.load("Y_test11.npy")

#Mischen der Menge von Training
arr_0 = np.arange(len(X_train))
np.random.shuffle(arr_0)
X_train = X_train[arr_0]
Y_train = Y_train[arr_0]

#Mischen der Menge von Test
arr_1 = np.arange(len(X_test))
np.random.shuffle(arr_1)
X_test = X_test[arr_1]
Y_test = Y_test[arr_1]

#Definition initialer Gewichte mit abgeschnittenen Normalverteilungen. Die Standardabweichung wird auf 0,1 gesetzt
def weight_variable(shape) :
    initial = tf.compat.v1.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

#Definition initialer Bias mit Null
def bias_variable(shape) :
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

#Strukturierung von Input
x = tf.compat.v1.placeholder(dtype=tf.float32,shape = [None,208,208,1])
y = tf.compat.v1.placeholder(dtype=tf.float32,shape = [None,4])

#Definition von Parameters
n_epochs = 10
batch_size = 50

#Definition erster Schichten von Faltung und Pooling
W_conv1 = weight_variable([7,7,1,64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1,2,2,1], padding="SAME") + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

#erster Restblock
#Definition zweiter Schichten von Faltung
W_conv2 = weight_variable([3,3,64,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)
#Definition dritter Schichten von Faltung
W_conv3 = weight_variable([3,3,64,64])
b_conv3 = bias_variable([64])
identity1 = h_pool1
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1,1,1,1], padding="SAME") + b_conv3+ identity1)

#zweiter Restblock
#Definition vierter Schichten von Faltung
W_conv4 = weight_variable([3,3,64,128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1,2,2,1], padding="SAME") + b_conv4)
#Definition fünfter Schichten von Faltung
W_conv5 = weight_variable([3,3,128,128])
b_conv5 = bias_variable([128])
W_conv_identity2 = weight_variable([1,1,64,128])
identity2 = tf.nn.conv2d(h_conv3, W_conv_identity2, strides=[1,2,2,1], padding="SAME")
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1,1,1,1], padding="SAME") + b_conv5+ identity2)

#dritter Restblock
#Definition sechster Schichten von Faltung
W_conv6 = weight_variable([3,3,128,256])
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=[1,2,2,1], padding="SAME") + b_conv6)
#Definition siebter Schichten von Faltung
W_conv7 = weight_variable([3,3,256,256])
b_conv7 = bias_variable([256])
W_conv_identity3 = weight_variable([1,1,128,256])
identity3 = tf.nn.conv2d(h_conv5, W_conv_identity3, strides=[1,2,2,1], padding="SAME")
h_conv7 = tf.nn.relu(tf.nn.conv2d(h_conv6, W_conv7, strides=[1,1,1,1], padding="SAME") + b_conv7+ identity3)

#vierter Restblock
#Definition achter Schichten von Faltung
W_conv8 = weight_variable([3,3,256,256])
b_conv8 = bias_variable([256])
h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, W_conv8, strides=[1,2,2,1], padding="SAME") + b_conv8)
#Definition neunter Schichten von Faltung
W_conv9 = weight_variable([3,3,256,256])
b_conv9 = bias_variable([256])
W_conv_identity4 = weight_variable([1,1,256,256])
identity4 = tf.nn.conv2d(h_conv7, W_conv_identity4, strides=[1,2,2,1], padding="SAME")
h_conv9 = tf.nn.relu(tf.nn.conv2d(h_conv8, W_conv9, strides=[1,1,1,1], padding="SAME") + b_conv9+ identity4)

#zweiter Schichten von Pooling
h_pool2 = tf.nn.avg_pool(h_conv9,ksize=[1,7,7,1], strides=[1,1,1,1], padding="VALID")

#Definition erster vollständig verbundene Schicht
W_fc1 = weight_variable([256*1*1, 1000])
b_fc1 = bias_variable([1000])
h_pool2_reshape = tf.reshape(h_pool2, [-1,1*1*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, W_fc1) + b_fc1)

#Definition von SoftMax-Schicht zur Implementierung der Klassifikation
W_fcS = weight_variable([1000,4])
b_fcS = bias_variable([4])
y_pred = tf.nn.softmax(tf.matmul(h_fc1, W_fcS) + b_fcS)

#Definition der Verlustfunktion und Bewertungsfunktion
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(y_pred), axis=1))
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Bilden der Initialisierung
init = tf.compat.v1.global_variables_initializer()

#Objekt für Modell-Speicherung
saver = tf.compat.v1.train.Saver()

#Trainieren
with tf.compat.v1.Session() as sess :
    init.run()
    #saver.restore(sess,"./my_model2.ckpt")

    n_batch = X_train.shape[0] // batch_size
    for epoch in range(n_epochs):
        print(epoch)
        for i in range(n_batch):
            train_step.run(feed_dict={x: X_train[i * batch_size: i * batch_size + batch_size], y: Y_train[i * batch_size: i * batch_size + batch_size]})
        if epoch != 0 and epoch%50 == 0:
            save_path = saver.save(sess, "./my_model2.ckpt")
    train_acc = accuracy.eval(feed_dict={x: X_train[:500], y: Y_train[:500]})
    test_acc = accuracy.eval(feed_dict={x: X_test[:500], y: Y_test[:500]})

print("train_acc:"+str(train_acc))
print("test_acc:" +str(test_acc))

