import cv2
import numpy as np
import pdb
import glob
import os
path='./tmp/debug_infer/'
txts=glob.glob(path+'Task1*')
for i_txt in range(len(txts)):
    if i_txt<3:
        continue
    txt=txts[i_txt]
    with open(txt) as f:
        data=f.readlines()

    im_name=[var.split(' ')[0] for var in data]
    im_name=list(set(im_name))
    im_dict={}
    for name in im_name:
        im_dict[name]=[]
    for var in data:
        var =var.strip()
        try:
            im_dict[var.split(' ')[0]].append(np.array(var.split(' ')[2:],dtype=np.float32))
        except:
            pdb.set_trace()
        
    for im in im_name[::56]:
        im_draw=cv2.imread('/disk1/yxzhu/data/icpr2018/DOTA_TEST/test/images/images/'+im+'.png')
        #pdb.set_trace()
        im_draw=im_draw.copy()
        polys=np.array(im_dict[im])
        for poly in polys:

           cv2.polylines(im_draw,[poly.astype(np.int32).reshape([-1,1,2])],True,(0,0,255),3)
        if not os.path.exists('./tmp/see_result/'+os.path.basename(txt)[:-4]):
            os.makedirs('./tmp/see_result/'+os.path.basename(txt)[:-4])
        cv2.imwrite('./tmp/see_result/'+os.path.basename(txt)[:-4]+'/'+im+'.jpg',im_draw)

    #pdb.set_trace()

