'''
    get ground truth bbox info, including areas, aspect ratio, draw a hist map
'''
import os 
import shapely.geometry as geom 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
from math import *

anno_file_root_path = r'/home/yxzh/dataset/icpr2018/train_split_cxma/labelTxt'
name_list = r'/home/yxzh/dataset/icpr2018/train_split_cxma/name_list.txt'
f = open(name_list,'r')
all_box={}
all_box['area']=[]
all_box['aspect_ratio'] = []
filenames = f.readlines()
for filename in filenames[:]:
    file = os.path.join(anno_file_root_path,filename.strip()+'.txt')
    boxes = open(file,'r')
    boxes = boxes.readlines()
    for box in boxes:
        box = box.strip().split(' ')
        box = box[:8]
        x1 = int(float(box[0]))
        y1 = int(float(box[1]))
        x2 = int(float(box[2]))
        y2 = int(float(box[3]))
        x3 = int(float(box[4]))
        y3 = int(float(box[5]))
        x4 = int(float(box[6]))
        y4 = int(float(box[7]))
        left = min(x1,x2,x3,x4)
        right = max(x1,x2,x3,x4)
        top = min(y1,y2,y3,y4)
        bottom = max(y1,y2,y3,y4)
        width = right - left + 1
        height = bottom - top + 1
        #print left, right, top, bottom
        #pdb.set_trace()

        #box = geom.Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
        #if not (box.is_valid and right-left>1 and bottom-top>1):
        #    continue
        print left, right, top, bottom
        if  width < 2: 
            continue
        if  height < 2: 
            continue
        all_box['area'].append(sqrt(width * height))
        all_box['aspect_ratio'].append(width * 1.0 / height)

print all_box['area'][:5],len(all_box['area'])
print all_box['aspect_ratio'][:5]
#plt.figure(1)
plt.subplot(1,2,1)
df = pd.DataFrame((np.array(all_box['area'])[:,np.newaxis]),columns=['Area'])
df.boxplot()
plt.subplot(1,2,2)
df2 = pd.DataFrame((np.array(all_box['aspect_ratio'])[:,np.newaxis]),columns=['Aspect_ratio'])
df2.boxplot()

plt.figure()
fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
ax0.hist(sorted(np.array(all_box['area']))[:-1000],500,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
ax0.set_title('area')
ax1.hist(sorted(np.array(all_box['aspect_ratio']))[:-1000],500,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
ax1.set_title('aspect_ratio')
plt.show()

