from random import shuffle
import numpy as np
import h5py
import cv2
import glob
import matplotlib.pyplot as plt
import cv
import time
import os
cat_dog_train_path = '../data/imagenet_data/val_raw/*'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
print (addrs)
addrs = [x for x in addrs if x[-9] == 'n']
print len(addrs), addrs[0]

sum = [0]
for i,x in enumerate(addrs):
    cat_dog_train_path = x + '/*'
    # read addresses and labels from the 'train' folder
    temp = glob.glob(cat_dog_train_path)
    sum[0] = sum[0] + len(temp)
    print sum[0]
    for y in temp:
        #print (y)
        #X_train_addrs = X_train_addrs + temp
        #Y_train_labels = Y_train_labels + list(np.full((len(temp)), class_to_idx[x]))
        #print y.split('/')[-2], '../val/'+ x + '/' + y.split('/')[-1]
        #break
        #print 'imagenet_data/val/'+ y.split('/')[-2] + '/' + y.split('/')[-1]
        img = cv2.imread(y)
        #plt.imshow(img)
        #plt.show()
        #img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        #print('../data/imagenet_data/val/'+ y.split('/')[-2] + '/' + y.split('/')[-1])
        cv2.imwrite('../data/imagenet_data/val/'+ y.split('/')[-2] + '/' + y.split('/')[-1],img)
