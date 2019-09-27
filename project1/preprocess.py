#
#
#
#
import os
import cv2
import numpy as np

def generate_face_pic(path,data):
    x_axis=int(data[1]*4/3)
    y_axis=int(data[0]*4/3)
    center_x=int(data[3])
    center_y=int(data[4])
    img=cv2.imread(path+'.jpg')
    # img2=img[(center_x-x_axis):(center_x+x_axis)][(center_y-y_axis):(center_y+y_axis)][:]
    sum_rows=img.shape[0]   #height
    sum_cols=img.shape[1]   #width
    # cv2.imwrite('image.png', img2)
    name=path.replace('/','_')

    img1 = np.zeros((y_axis*2,x_axis*2,3), np.uint8)
    img1.fill(255)
    for i in range(y_axis*2):
        for j in range(x_axis*2):
            for k in range(3):
                img1[i][j][k]=img[(center_y-y_axis+i)%(sum_rows)][(center_x-x_axis+j)%(sum_cols)][k]
    save_path='train/rawclip/'
    cv2.imwrite(save_path+name+'.jpg', img1)
    with open('train/bblist.txt', 'a') as f:
        f.write(save_path+name+'.jpg\n')

    square_pic=cv2.resize(img1,(96,96))
    positive_path='train/positive/'
    negative_path='train/negative/'

    cropped=square_pic[12:84,12:84]
    cv2.imwrite(positive_path+name+'.jpg', cropped)
    with open('train/poslist.txt', 'a') as f:
        f.write(positive_path+name+'.jpg\n') 

    cropped=square_pic[0:72,0:72]
    cv2.imwrite(negative_path+name+'_1.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_1.jpg\n') 

    cropped=square_pic[12:84,0:72]
    cv2.imwrite(negative_path+name+'_2.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_2.jpg\n')

    cropped=square_pic[24:96,0:72]
    cv2.imwrite(negative_path+name+'_3.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_3.jpg\n')        

    cropped=square_pic[0:72,12:84]
    cv2.imwrite(negative_path+name+'_4.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_4.jpg\n')
    
    cropped=square_pic[24:96,12:84]
    cv2.imwrite(negative_path+name+'_5.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_5.jpg\n')
    
    cropped=square_pic[0:72,24:96]
    cv2.imwrite(negative_path+name+'_6.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_6.jpg\n')
    
    cropped=square_pic[12:84,24:96]
    cv2.imwrite(negative_path+name+'_7.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_7.jpg\n')

    cropped=square_pic[12:84,24:96]
    cv2.imwrite(negative_path+name+'_8.jpg', cropped)
    with open('train/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_8.jpg\n')

    return

def generate_face_pic_test(path,data):
    x_axis=int(data[1]*4/3)
    y_axis=int(data[0]*4/3)
    center_x=int(data[3])
    center_y=int(data[4])
    img=cv2.imread(path+'.jpg')
    # img2=img[(center_x-x_axis):(center_x+x_axis)][(center_y-y_axis):(center_y+y_axis)][:]
    sum_rows=img.shape[0]   #height
    sum_cols=img.shape[1]   #width
    # cv2.imwrite('image.png', img2)
    name=path.replace('/','_')

    img1 = np.zeros((y_axis*2,x_axis*2,3), np.uint8)
    img1.fill(255)
    for i in range(y_axis*2):
        for j in range(x_axis*2):
            for k in range(3):
                img1[i][j][k]=img[(center_y-y_axis+i)%(sum_rows)][(center_x-x_axis+j)%(sum_cols)][k]
    save_path='test/rawclip/'
    cv2.imwrite(save_path+name+'.jpg', img1)
    with open('test/bblist.txt', 'a') as f:
        f.write(save_path+name+'.jpg\n')

    square_pic=cv2.resize(img1,(96,96))
    positive_path='test/positive/'
    negative_path='test/negative/'

    cropped=square_pic[12:84,12:84]
    cv2.imwrite(positive_path+name+'.jpg', cropped)
    with open('test/poslist.txt', 'a') as f:
        f.write(positive_path+name+'.jpg\n') 

    cropped=square_pic[0:72,0:72]
    cv2.imwrite(negative_path+name+'_1.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_1.jpg\n') 

    cropped=square_pic[12:84,0:72]
    cv2.imwrite(negative_path+name+'_2.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_2.jpg\n')

    cropped=square_pic[24:96,0:72]
    cv2.imwrite(negative_path+name+'_3.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_3.jpg\n')        

    cropped=square_pic[0:72,12:84]
    cv2.imwrite(negative_path+name+'_4.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_4.jpg\n')
    
    cropped=square_pic[24:96,12:84]
    cv2.imwrite(negative_path+name+'_5.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_5.jpg\n')
    
    cropped=square_pic[0:72,24:96]
    cv2.imwrite(negative_path+name+'_6.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_6.jpg\n')
    
    cropped=square_pic[12:84,24:96]
    cv2.imwrite(negative_path+name+'_7.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_7.jpg\n')

    cropped=square_pic[12:84,24:96]
    cv2.imwrite(negative_path+name+'_8.jpg', cropped)
    with open('test/neglist.txt', 'a') as f:
        f.write(negative_path+name+'_8.jpg\n')

    return

# first 8 folders are training samples, last 2 folders

for item in range(1,8):
    path_simple='FDDB-folds/FDDB-fold-0'+str(item)+'.txt'
    fo=open(path_simple)
    pic_name=fo.readlines() #list, len(list)
    fo.close()
    for i in range(0, len(pic_name)):
        pic_name[i] = pic_name[i].rstrip('\n')
    pic_num=len(pic_name)

    path_full='FDDB-folds/FDDB-fold-0'+str(item)+'-ellipseList.txt'

    fo=open(path_full)
    for i in range(pic_num):
        name=fo.readline().rstrip('\n')
        num=int(fo.readline().rstrip('\n'))
        for j in range(num):
            data_line=fo.readline().rstrip('\n')
            facedata_str = list(filter(None,data_line.split(" ")))
            facedata=[float(s) for s in facedata_str]
            #<major_axis_radius minor_axis_radius angle center_x center_y detection_score>
            generate_face_pic(name,facedata)

    fo.close()

path_simple='FDDB-folds/FDDB-fold-09.txt'
fo=open(path_simple)
pic_name=fo.readlines() #list, len(list)
fo.close()
for i in range(0, len(pic_name)):
    pic_name[i] = pic_name[i].rstrip('\n')
pic_num=len(pic_name)

path_full='FDDB-folds/FDDB-fold-09-ellipseList.txt'

fo=open(path_full)
for i in range(pic_num):
    name=fo.readline().rstrip('\n')
    num=int(fo.readline().rstrip('\n'))
    for j in range(num):
        data_line=fo.readline().rstrip('\n')
        facedata_str = list(filter(None,data_line.split(" ")))
        facedata=[float(s) for s in facedata_str]
        #<major_axis_radius minor_axis_radius angle center_x center_y detection_score>
        generate_face_pic_test(name,facedata)

fo.close()

path_simple='FDDB-folds/FDDB-fold-10.txt'
fo=open(path_simple)
pic_name=fo.readlines() #list, len(list)
fo.close()
for i in range(0, len(pic_name)):
    pic_name[i] = pic_name[i].rstrip('\n')
pic_num=len(pic_name)

path_full='FDDB-folds/FDDB-fold-10-ellipseList.txt'

fo=open(path_full)
for i in range(pic_num):
    name=fo.readline().rstrip('\n')
    num=int(fo.readline().rstrip('\n'))
    for j in range(num):
        data_line=fo.readline().rstrip('\n')
        facedata_str = list(filter(None,data_line.split(" ")))
        facedata=[float(s) for s in facedata_str]
        #<major_axis_radius minor_axis_radius angle center_x center_y detection_score>
        generate_face_pic_test(name,facedata)

fo.close()


print("hh")