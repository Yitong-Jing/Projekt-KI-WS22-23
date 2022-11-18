import tensorflow as tf
import os
import numpy as np

d_name = os.listdir("D:/projekt/Alzheimer/train/")
train_path = "D:/projekt/Alzheimer/test/"

images = []
images_labels =[]

for i in range(len(d_name)) :
    path_list = tf.data.Dataset.list_files(train_path + d_name[i] + "/" + "*.jpg",shuffle=False)
    for j in path_list:
        print(j)
        image_temp = tf.io.read_file(j)
        image_temp = tf.image.decode_jpeg(image_temp)
        images.append(image_temp)
        images_labels.append(i,)

X, Y = np.array(images), np.array(images_labels)

np.save("X_test.npy",X)
np.save("Y_test.npy",Y)
print(X.shape)
print(X.dtype)
print(Y.shape)
print(Y.dtype)