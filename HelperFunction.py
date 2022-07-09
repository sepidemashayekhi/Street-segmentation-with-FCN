import os 
import cv2 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def ben(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    new_mask=new_mask.astype(np.float32)
    return new_mask

def getSegmentationArr(mask, classes, width, height):
    seg_label=np.zeros(shape=(height,width,classes))
    img=mask[:,:,0]
    for c in range(classes):
        seg_label[:,:,c]= (img == c ).astype(int)
    return seg_label

def give_color_to_seg_img(seg, n_classes):
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    # seg=np.reshape(seg,seg.shape[:2])
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))
    return seg_img

def DataGenerator(path,batch_size,classes):
    global image_path,mask_path
    image_path=path[0]
    mask_path=path[1]
    image_files=os.listdir(image_path)
    mask_files=os.listdir(mask_path)
    while True:
        for i in range(0,len(image_files),batch_size):
            image_batch_flile=image_files[i:i+batch_size]
            mask_batch_flile=mask_files[i:i+batch_size]
            imgs=[]
            segs=[]
            for file in zip(image_batch_flile,mask_batch_flile):
                image=cv2.imread(image_path+'/'+file[0])
                mask=cv2.imread(mask_path+'/'+file[1])
                new_mask=ben(mask)
                labels=getSegmentationArr(new_mask,classes)
                imgs.append(image)
                segs.append(labels)
            yield np.array(imgs), np.array(segs)
