#
#
#
#
import os
import numpy as np

from skimage.feature import hog
from skimage import io
import cv2
# import matplotlib.pyplot as plt



# path='train/positive/2002_07_19_big_img_230.jpg'
# img=cv2.imread(path)
# cv2.imwrite('test.jpg', img)
# nor1=hog(img,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),visualise=False)
# print(nor1)
# print(nor1.shape)
# print('hh')

# generate hog of positive and negative of training set
path_train_pos='train/poslist.txt'
path_save_train_pos='train/hog_pos.npy'
fo=open(path_train_pos)
pic_train_pos=fo.readlines()
fo.close()
for i in range(0, len(pic_train_pos)):
        pic_train_pos[i] = pic_train_pos[i].rstrip('\n')
pos_features=[]
for p in pic_train_pos:
    img=cv2.imread(p)
    feature=hog(img,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),visualise=False)
    pos_features.append(feature)

pos_features=np.array(pos_features)
np.save(path_save_train_pos, pos_features)
# np.load('save_x.npy')


path_train_neg='train/neglist.txt'
path_save_train_neg='train/hog_neg.npy'
fo=open(path_train_neg)
pic_train_neg=fo.readlines()
fo.close()
for i in range(0, len(pic_train_neg)):
        pic_train_neg[i] = pic_train_neg[i].rstrip('\n')
neg_features=[]
for p in pic_train_neg:
    img=cv2.imread(p)
    feature=hog(img,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),visualise=False)
    neg_features.append(feature)

neg_features=np.array(neg_features)
np.save(path_save_train_neg, neg_features)
# np.load('save_x.npy')



path_test_pos='test/poslist.txt'
path_save_test_pos='test/hog_pos.npy'
fo=open(path_test_pos)
pic_test_pos=fo.readlines()
fo.close()
for i in range(0, len(pic_test_pos)):
        pic_test_pos[i] = pic_test_pos[i].rstrip('\n')
pos_features=[]
for p in pic_test_pos:
    img=cv2.imread(p)
    feature=hog(img,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),visualise=False)
    pos_features.append(feature)

pos_features=np.array(pos_features)
np.save(path_save_test_pos, pos_features)


path_test_neg='test/neglist.txt'
path_save_test_neg='test/hog_neg.npy'
fo=open(path_test_neg)
pic_test_neg=fo.readlines()
fo.close()
for i in range(0, len(pic_test_neg)):
        pic_test_neg[i] = pic_test_neg[i].rstrip('\n')
neg_features=[]
for p in pic_test_neg:
    img=cv2.imread(p)
    feature=hog(img,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),visualise=False)
    neg_features.append(feature)

neg_features=np.array(neg_features)
np.save(path_save_test_neg, neg_features)