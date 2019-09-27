import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import cv2

def shuffle(X,Y):
    np.random.seed(2)
    randomList=np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

path_train_pos='train/poslist.txt'
path_train_neg='train/neglist.txt'
path_test_pos='test/poslist.txt'
path_test_neg='test/neglist.txt'

x_pos=[]

fo=open(path_train_pos)
pic_train_pos=fo.readlines()
fo.close()
for i in range(0, len(pic_train_pos)):
    pic_train_pos[i] = pic_train_pos[i].rstrip('\n')

for p in pic_train_pos:
    img=cv2.imread(p)
    x_pos.append(img)

fo=open(path_train_neg)
pic_train_neg=fo.readlines()
fo.close()
for i in range(0, len(pic_train_neg)):
    pic_train_neg[i] = pic_train_neg[i].rstrip('\n')

for p in pic_train_neg:
    img=cv2.imread(p)
    x_pos.append(img)

pos_num=len(pic_train_pos)
neg_num=len(pic_train_neg)

x = np.ones(pos_num, dtype = int)
y = np.zeros(neg_num, dtype = int)
label_num=np.hstack((x,y))

label=keras.utils.to_categorical(label_num,num_classes=2)

x_pos=np.array(x_pos)

x_train, y_train = shuffle(x_pos, label)

#############


print(x_pos.shape)

##############
x_pos=[]

fo=open(path_test_pos)
pic_test_pos=fo.readlines()
fo.close()
for i in range(0, len(pic_test_pos)):
    pic_test_pos[i] = pic_test_pos[i].rstrip('\n')

for p in pic_test_pos:
    img=cv2.imread(p)
    x_pos.append(img)

fo=open(path_test_neg)
pic_test_neg=fo.readlines()
fo.close()
for i in range(0, len(pic_test_neg)):
    pic_test_neg[i] = pic_test_neg[i].rstrip('\n')

for p in pic_test_neg:
    img=cv2.imread(p)
    x_pos.append(img)

pos_num=len(pic_test_pos)
neg_num=len(pic_test_neg)

x = np.ones(pos_num, dtype = int)
y = np.zeros(neg_num, dtype = int)
label_num=np.hstack((x,y))

label=keras.utils.to_categorical(label_num,num_classes=2)

x_pos=np.array(x_pos)

x_test, y_test = shuffle(x_pos, label)



model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(72, 72, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=1)
score, acc = model.evaluate(x_test, y_test, batch_size=32)

print(score)
print('\n')
print(acc)