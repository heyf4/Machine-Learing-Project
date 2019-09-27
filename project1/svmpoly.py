#
#
#
#
import os
import numpy as np

from skimage.feature import hog
from skimage import io
import cv2

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

#load pos train

ft_pos=np.load('train/hog_pos.npy')

ft_neg=np.load('train/hog_neg.npy')

model=svm.SVC(kernel = 'poly')

x = np.ones([ft_pos.shape[0],1], dtype = int)
data_pos=np.hstack((ft_pos,x))
print(data_pos.shape)
y = -1*np.ones([ft_neg.shape[0],1], dtype = int)
data_neg=np.hstack((ft_neg,y))
print(data_neg.shape)
data=np.vstack((data_pos,data_neg))
print(data.shape)
np.random.shuffle(data)


x_train=data[:,:-1]
y_train=data[:,-1]

model.fit(x_train,y_train)


ft_pos=np.load('test/hog_pos.npy')

ft_neg=np.load('test/hog_neg.npy')

x = np.ones([ft_pos.shape[0],1], dtype = int)
data_pos=np.hstack((ft_pos,x))
print(data_pos.shape)
y = -1*np.ones([ft_neg.shape[0],1], dtype = int)
data_neg=np.hstack((ft_neg,y))
print(data_neg.shape)
data=np.vstack((data_pos,data_neg))
print(data.shape)
# np.random.shuffle(data)

x_test=data[:,:-1]
y_test=data[:,-1]

y_prediction=model.predict(x_test)

print('accuracy'+str(accuracy_score(y_test,y_prediction))+'\n')
print(classification_report(y_test,y_prediction))
print('hh')