# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import shapely.geometry
from shapely.geometry import Polygon
from shapely.geometry import LinearRing
import copy
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import cv2
import pdb
# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from core.config import cfg
from datasets.dataset_catalog import ANN_FN
from datasets.dataset_catalog import ANN_DIR
from datasets.dataset_catalog import NAMELIST
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import IM_DIR
from datasets.dataset_catalog import IM_PREFIX
from utils.timer import Timer
import utils.boxes as box_utils
#pdb.set_trace()
logger = logging.getLogger(__name__)

def rotate_bound(shape, angle,cxy=None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = shape
    #pdb.set_trace()
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    #
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return M

def rotate(im_shape,text_polys,rd_rotate):

    #random_rotate=np.array([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
    #rd_rotate=np.random.choice(random_rotate)

    m=rotate_bound(im_shape,rd_rotate)
    #pdb.set_trace()
    text_polys=text_polys.reshape([len(text_polys),-1,2])
    text_polys=np.concatenate((text_polys,np.ones([len(text_polys),text_polys.shape[1],1],dtype=np.float32)),-1)
    m=np.transpose(m,[1,0])
    text_polys=np.dot(text_polys,m.astype(np.float32))
    text_polys=text_polys.reshape(len(text_polys),-1)
    return text_polys

class TxtDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_DIR]), \
            'Annotation directory \'{}\' not found'.format(DATASETS[name][ANN_DIR])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.annotation_directory = DATASETS[name][ANN_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        #self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        self.cls_names = ['__background__',\
                        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', \
                        'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  \
                        'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
        # Set up dataset classes
        #category_ids = self.COCO.getCatIds()
        #categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        category_ids = [i+1 for i in range(len(self.cls_names))]
        categories = self.cls_names[1:]

        self.category_to_id_map = dict(zip(categories, category_ids))
        #print(self.category_to_id_map)
        #pdb.set_trace()
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(category_ids)
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()
        #self.dataset_size = int(DATASETS[name][SIZE])
        self.namelist = DATASETS[name][NAMELIST]


    def get_roidb(
        self,
        gt=False,
        proposal_file=None,
        min_proposal_size=2,
        proposal_limit=-1,
        crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        #image_ids = self.COCO.getImgIds()
        namelist = open(self.namelist,'r')
        names = namelist.readlines()

        image_ids = [i for i in range(len(names))]

        #image_ids.sort()
        self.dataset_size = len(image_ids)
        #roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        

        #'''this is code for rotation augment
        roidb_lt=[]
        
        for angle in [0,90,180,270]:
            
            roidb = [{} for i in range(self.dataset_size)]
            roidb_lt.append(roidb)
            #for idx, entry in zip(image_ids,roidb):
            for idx, name, entry in zip(image_ids,names,roidb):
                suffix = '.jpg'
                entry['id'] = idx
                entry['file_name'] = name.strip() + suffix
                self._prep_roidb_entry(entry,angle)
            #with open('roi.pik', 'w') as f:
                #entry=pickle.dump(f)
            if gt:
                # Include ground-truth object annotations

                self.debug_timer.tic()
                #pdb.set_trace()
                for entry in roidb:
                    self._add_gt_annotations_rotation(entry,angle)
                #pdb.set_trace()
                logger.debug(
                    '_add_gt_annotations_rotation took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
        var_all=[]
        [var_all.extend(var) for var in roidb_lt]
        roidb=var_all
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)

        #'''

        '''this is code for roidb without rotation augment
        roidb = [{} for i in range(self.dataset_size)]
        for idx, name, entry in zip(image_ids,names,roidb):
            suffix = '.jpg'
            entry['id'] = idx
            entry['file_name'] = name.strip() + suffix
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            self.debug_timer.tic()
            for entry in roidb:
                self._add_gt_annotations(entry)
            #pdb.set_trace()
            logger.debug(
                '_add_gt_annotations took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        '''

        #class balance 
        samples=np.zeros(cfg.MODEL.NUM_CLASSES-1).astype(np.int32)
        roi_lt=[[] for i in range(cfg.MODEL.NUM_CLASSES-1)]

        for roidb_one in roidb:
            num_one=np.zeros(cfg.MODEL.NUM_CLASSES-1).astype(np.int32)
            sample_one=np.zeros(cfg.MODEL.NUM_CLASSES-1).astype(np.int32)
            for i in range(cfg.MODEL.NUM_CLASSES-1):
                num_one[i]=np.sum(((roidb_one['gt_classes']==i+1)&(roidb_one['is_crowd']==0)).astype(np.int32))
            #sample_one[i]
            if (num_one==0).all():
                continue
            sample_one[num_one.argmax()]=1
            roi_lt[num_one.argmax()].append(roidb_one)
            #sample_one[num_one/float(len(num_one))>0.3]=1
            samples+=sample_one
        
        max_num=np.max(samples)
        
        #
        for i in range(cfg.MODEL.NUM_CLASSES-1):
            for j in range(int(np.round((max_num-samples[i])/float(len(roi_lt[i]))))):
                roidb+=roi_lt[i]
        random.shuffle(roidb)
        #print(max_num)
        #pdb.set_trace()
        return roidb

    def _prep_roidb_entry(self, entry, angle):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        ann_path = os.path.join(self.annotation_directory, entry['file_name'][:-4]+'.txt')
        assert os.path.exists(ann_path), 'Annotation \'{}\' not found'.format(ann_path)
        entry['annotation'] = ann_path
        entry['flipped'] = False
        entry['angle']=angle
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['polyinters'] = np.empty((0, 28), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]
        #read im to get its width and height
        im = cv2.imread(entry['image'])
        #print("Reading image:{}\n".format(entry['image']))
        logger.info("Reading image:{}\n".format(entry['image']))
        entry['width'] = im.shape[1]
        entry['height'] = im.shape[0]


    def _add_gt_annotations_rotation(self, entry, angle=0.):
        """Add ground truth annotation metadata to an roidb entry."""
        #ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        #objs = self.COCO.loadAnns(ann_ids)
        ann_file = open(entry['annotation'],'r')

        objs = ann_file.readlines()
        
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            try:
                obj = obj.strip().split(' ')
                assert len(obj) == 10, "There is an error in the annotation file: {}".format(entry['ann_path'])
                obj_dict = {}
                obj_dict['category_id'] = self.category_to_id_map[obj[8]]
                assert (int(obj[9]) in [0,1,2])
                obj_dict['difficult'] = int(obj[9]) #0 if obj[9] == '0' else 1 # 2 for outer box when split; 1 for difficult; 0 for not difficult
                obj_dict['iscrowd'] = 0 # this attribute is from coco, set it to be 0 for default
                obj_dict['segmentation'] = [[float(i) for i in obj[:8]]]
                obj_dict['area'] = calcArea(obj_dict['segmentation'][0])
                if obj_dict['area'] < cfg.TRAIN.GT_MIN_AREA:
                    continue
                #if obj_dict['difficult'] == 1 and cfg.TRAIN.SKIP_DIFFICULT_OBJ:
                #    continue
                if obj_dict['difficult'] == 2:
                    continue
                if 'ignore' in obj_dict and obj_dict['ignore'] == 1:
                    continue
                p_x1,p_y1,p_x2,p_y2,p_x3,p_y3,p_x4,p_y4=obj_dict['segmentation'][0]
                plg = Polygon([(obj_dict['segmentation'][0][0],obj_dict['segmentation'][0][1]),\
                            (obj_dict['segmentation'][0][2],obj_dict['segmentation'][0][3]),\
                            (obj_dict['segmentation'][0][4],obj_dict['segmentation'][0][5]),\
                            (obj_dict['segmentation'][0][6],obj_dict['segmentation'][0][7])])
                if not plg.is_valid: 
                    continue
                polys_gt=rotate((height,width),np.array([[p_x1,p_y1,p_x2,p_y2,p_x3,p_y3,p_x4,p_y4]],dtype=np.float32),angle)
                x1 = np.min(polys_gt[0][0::2])
                x2 = np.max(polys_gt[0][0::2])
                y1 = np.min(polys_gt[0][1::2])
                y2 = np.max(polys_gt[0][1::2])
                p_x1,p_y1,p_x2,p_y2,p_x3,p_y3,p_x4,p_y4=polys_gt[0]
                obj_dict['bbox'] = [x1,y1,x2-x1+1,y2-y1+1]
                
                # Require non-zero seg area and more than 1x1 box size
                if obj_dict['area'] > 0 and x2 > x1 and y2 > y1:
                    obj_dict['clean_bbox'] = [x1, y1, x2, y2]
                    #obj_dict['clean_polyinter']=xy_inter
                    valid_objs.append(copy.deepcopy(obj_dict))
                    valid_segms.append(copy.deepcopy(obj_dict['segmentation']))
                    #pdb.set_trace()
            except:
                pdb.set_trace()
                print('wrong')
                continue
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        #polyinters=np.zeros((num_valid_objs, 28), dtype=entry['polyinters'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )


        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            #polyinters[ix,:]=obj['clean_polyinter']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']


            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        #entry['polyinters']=np.append(entry['polyinters'],polyinters,axis=0)
        #pdb.set_trace()
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        #ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        #objs = self.COCO.loadAnns(ann_ids)
        ann_file = open(entry['annotation'],'r')
        objs = ann_file.readlines()
        
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            obj = obj.strip().split(' ')
            assert len(obj) == 10, "There is an error in the annotation file: {}".format(entry['ann_path'])
            obj_dict = {}
            obj_dict['category_id'] = self.category_to_id_map[obj[8]]
            assert (int(obj[9]) in [0,1,2])
            obj_dict['difficult'] = int(obj[9]) #0 if obj[9] == '0' else 1 # 2 for outer box when split; 1 for difficult; 0 for not difficult
            obj_dict['iscrowd'] = 0 # this attribute is from coco, set it to be 0 for default
            obj_dict['segmentation'] = [[int(float(i)) for i in obj[:8]]]
            obj_dict['area'] = calcArea(obj_dict['segmentation'][0])
            if obj_dict['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            #if obj_dict['difficult'] == 1 and cfg.TRAIN.SKIP_DIFFICULT_OBJ:
            #    continue
            if obj_dict['difficult'] == 2:
                continue
            if 'ignore' in obj_dict and obj_dict['ignore'] == 1:
                continue
            plg = Polygon([(obj_dict['segmentation'][0][0],obj_dict['segmentation'][0][1]),\
                        (obj_dict['segmentation'][0][2],obj_dict['segmentation'][0][3]),\
                        (obj_dict['segmentation'][0][4],obj_dict['segmentation'][0][5]),\
                        (obj_dict['segmentation'][0][6],obj_dict['segmentation'][0][7])])
            if not plg.is_valid: 
                continue
            x1 = np.min(obj_dict['segmentation'][0][0::2])
            x2 = np.max(obj_dict['segmentation'][0][0::2])
            y1 = np.min(obj_dict['segmentation'][0][1::2])
            y2 = np.max(obj_dict['segmentation'][0][1::2])
            '''
            line=np.arange(0,1,1/8.)
            line=line[1:]
            #line=line.reshape(1,len(line))
            dx=x2-x1
            dy=y2-y1
            line_x=dx*line+x1
            line_y=dy*line+y1
            p_x1,p_y1,p_x2,p_y2,p_x3,p_y3,p_x4,p_y4=obj_dict['segmentation'][0]

            p_x1, p_y1, p_x2, p_y2 = box_utils.clip_xyxy_to_image(
                p_x1, p_y1, p_x2, p_y2, height, width
            )
            p_x3, p_y3, p_x4, p_y4 = box_utils.clip_xyxy_to_image(
                p_x3, p_y3, p_x4, p_y4, height, width
            )

            
            polygon = np.array([p_x1,p_y1,p_x2,p_y2,p_x3,p_y3,p_x4,p_y4],dtype=np.float32).reshape([-1,2])
            shapely_poly = LinearRing(polygon)
            inter_lt_x=[]
            inter_lt_y=[]
            for x in line_x:
                line = [(x, -1000000.), (x, +100000000.)]
                shapely_line = shapely.geometry.LineString(line)
                inter=shapely_poly.intersection(shapely_line)
                k=[var.xy for var in inter]
                x=[np.array(var[0]) for var in k]
                x=np.concatenate(x)[:,np.newaxis]
                y=[np.array(var[1]) for var in k]
                y=np.concatenate(y)[:,np.newaxis]
                intersection_line=np.concatenate((x,y),-1)
                idx_sort=np.argsort(intersection_line[:,1])
                #pdb.set_trace()
                intersection_line=[intersection_line[idx_sort[0]],intersection_line[idx_sort[-1]]]

                assert len(intersection_line)==2
                inter_lt_x.append(intersection_line)
            for y in line_y:
                line = [(-1000000.,y ), (+100000000.,y )]
                shapely_line = shapely.geometry.LineString(line)
                inter=shapely_poly.intersection(shapely_line)
                k=[var.xy for var in inter]
                x=[np.array(var[0]) for var in k]
                x=np.concatenate(x)[:,np.newaxis]
                y=[np.array(var[1]) for var in k]
                y=np.concatenate(y)[:,np.newaxis]
                intersection_line=np.concatenate((x,y),-1)
                idx_sort=np.argsort(intersection_line[:,0])
                #pdb.set_trace()
                intersection_line=[intersection_line[idx_sort[0]],intersection_line[idx_sort[-1]]]

                assert len(intersection_line)==2
                inter_lt_y.append(intersection_line)
            x_inter=np.array(inter_lt_x).reshape(-1,2)[:,1]
            y_inter=np.array(inter_lt_y).reshape(-1,2)[:,0]
            inter_lt_x=np.array(inter_lt_x).reshape(-1,2).astype(np.int32)
            inter_lt_y=np.array(inter_lt_y).reshape(-1,2).astype(np.int32)

            #
            xy_inter=np.zeros(28,dtype=np.float32)
            #xy_inter=np.concatenate((x_inter,y_inter))
            xy_inter[::2]=y_inter
            xy_inter[1::2]=x_inter
            '''
            

            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            #x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            
            #x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
            #    x1, y1, x2, y2, height, width
            #)
            obj_dict['bbox'] = [x1,y1,x2-x1+1,y2-y1+1]
            #if (xy_inter[1::2]>y2).any() or (xy_inter[::2]>x2).any() or (xy_inter[1::2]<y1).any() or (xy_inter[::2]<x1).any():
            #    pdb.set_trace()
            #if abs(xy_inter[0]-596.19512939)<0.01:
                    #pdb.set_trace()
            #        pass
            #if (xy_inter=np.array([596.19512939,974.35418701,613.63415527,994.50732422,595.39025879,  954.70831299,  612.82928467,  995.01470947,594.58538818,  953.46325684,  612.02441406,995.52203369,593.78051758,953.97058105, 611.21948242,  996.02941895,592.97558594,  954.47796631,  610.41461182,  996.53674316,592.17071533,  954.98529053,  609.60974121,  995.29168701,591.36584473,  955.49267578,608.80487061, 975.64581299],dtype=np.float32)).all():
            #        pdb.set_trace()
            # Require non-zero seg area and more than 1x1 box size
            if obj_dict['area'] > 0 and x2 > x1 and y2 > y1:
                obj_dict['clean_bbox'] = [x1, y1, x2, y2]
                #obj_dict['clean_polyinter']=xy_inter
                valid_objs.append(copy.deepcopy(obj_dict))
                valid_segms.append(copy.deepcopy(obj_dict['segmentation']))
                #pdb.set_trace()
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        #polyinters=np.zeros((num_valid_objs, 28), dtype=entry['polyinters'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            #polyinters[ix,:]=obj['clean_polyinter']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        #entry['polyinters']=np.append(entry['polyinters'],polyinters,axis=0)
        #pdb.set_trace()
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps

def calcArea(points): 
    m1,m2,n1,n2,j1,j2,k1,k2 = \
        float(points[0]),float(points[1]),float(points[2]),float(points[3]),\
        float(points[4]),float(points[5]),float(points[6]),float(points[7])
    q1 = [m1,m2] 
    q2 = [n1,n2] 
    q3 = [j1,j2] 
    q4 = [k1,k2] 
    d12 = calcDistance(q1[0], q1[1], q2[0], q2[1]) 
    d23 = calcDistance(q2[0], q2[1], q3[0], q3[1]) 
    d34 = calcDistance(q3[0], q3[1], q4[0], q4[1]) 
    d41 = calcDistance(q4[0], q4[1], q1[0], q1[1]) 
    d24 = calcDistance(q2[0], q2[1], q4[0], q4[1]) 
    k1 = (d12+d41+d24)/2 
    k2 = (d23+d34+d24)/2 
    s1 = (k1*(k1-d12)*(k1-d41)*(k1-d24))**0.5 
    s2 = (k2*(k2-d23)*(k2-d34)*(k2-d24))**0.5 
    s = s1+s2 
    return s 
def calcDistance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        pdb.set_trace()
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()

    def get_roidb(
        self,
        gt=False,
        proposal_file=None,
        min_proposal_size=2,
        proposal_limit=-1,
        crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            self.debug_timer.tic()
            for entry in roidb:
                self._add_gt_annotations(entry)
            logger.debug(
                '_add_gt_annotations took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        #pdb.set_trace()
        #entry['polyinters']=np.append(entry['polyinters'],
        #    np.zeros((num_boxes,28), dtype=entry['polyinters'].dtype),axis=0
        #    )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]




