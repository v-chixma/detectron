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
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.segms as segm_utils
import pdb


class GenerateMaskRoIsOp(object):
    """Output object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, train):
        self._train = train

    def forward(self, inputs, outputs):
        cls_prob = inputs[0].data
        bbox_pred_odai = inputs[1].data
        #roi has format:(R,5) batch_ind, x1, y1, x2, y2
        rois = inputs[2].data

        #print(cls_prob.shape)
        cls_prob = cls_prob.reshape((-1,cfg.MODEL.NUM_CLASSES))
        roi_cls_inds = np.argmax(cls_prob, axis = 1)
        bbox_pred_odai = bbox_pred_odai.reshape((-1,4*cfg.MODEL.NUM_CLASSES))
        #print(roi_cls_inds)
        #pdb.set_trace()

        if self._train:
            roidb = blob_utils.deserialize(inputs[3].data)
            im_info = inputs[4].data
            fg_inds = np.where(roi_cls_inds>0)[0]
            #output blob for training
            mask_rois = None
            masks = None
            M = cfg.MRCNN.RESOLUTION

            if len(fg_inds) > 0:
                #print('fg_inds:',fg_inds)
                mask_class_labels = np.empty((0,1))
                masks = blob_utils.zeros((len(fg_inds), M**2), int32=True)
                count = 0 
                fg_rois = rois[fg_inds]
                

                rois_deltas = np.empty((0,4),dtype=bbox_pred_odai.dtype)
                for k in range(len(bbox_pred_odai)):
                    rois_deltas = np.vstack((rois_deltas, bbox_pred_odai[k, 4*roi_cls_inds[k]:4*(roi_cls_inds[k]+1)]))
                fg_rois_deltas = rois_deltas[fg_inds, :]
                

                for i in range(im_info.shape[0]):
                    batch_rois_inds = np.where(fg_rois[:,0] == i)[0]
                    #print(batch_rois_inds)
                    #
                    if len(batch_rois_inds) > 0:
                        batch_rois = fg_rois[batch_rois_inds]
                        im_scale = im_info[i,-1]
                        #batch_labels = roi_cls_inds[fg_inds][np.where(fg_rois[:,0] == i)]
                        batch_boxes = batch_rois[:,1:] / im_scale
                        batch_rois_deltas = fg_rois_deltas[np.where(fg_rois[:,0] == i)[0]]
                        # Transform fg_rois into proposals via bbox transformations
                        batch_mask_boxes = box_utils.bbox_transform(
                            batch_boxes, batch_rois_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
                        batch_mask_rois = batch_mask_boxes * im_scale
                        batch_mask_rois = np.hstack((batch_rois[:,0].reshape((-1,1)),batch_mask_rois))
                        if mask_rois == None:
                            mask_rois = np.copy(batch_mask_rois)
                        else:
                            mask_rois = np.vstack((mask_rois,batch_mask_rois))


                        entry = roidb[i]
                        #pdb.set_trace()
                        polys_gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
                        #pdb.set_trace()
                        polys_gt = [entry['segms'][ind] for ind in polys_gt_inds]
                        polys_gt_class = [entry['gt_classes'][ind] for ind in polys_gt_inds]
                        boxes_from_polys = segm_utils.polys_to_boxes(polys_gt)
                        #fg_inds = np.where(blobs['labels_int32'] > 0)[0]
                        #roi_has_mask = blobs['labels_int32'].copy()
                        #roi_has_mask[roi_has_mask > 0] = 1                                                

                        # Find overlap between all foreground rois and the bounding boxes
                        # enclosing each segmentation
                        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
                            batch_mask_boxes.astype(np.float32, copy=False),
                            boxes_from_polys.astype(np.float32, copy=False)
                        )

                        # Map from each fg rois to the index of the mask with highest overlap
                        # (measured by bbox overlap)
                        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)
                        #pdb.set_trace()
                        # Class labels for the mask rois
                        for tmp in range(len(fg_polys_inds)):
                            mask_class_labels = np.vstack((mask_class_labels, polys_gt_class[fg_polys_inds[tmp]]))

                        # add fg targets
                        for j in range(batch_mask_boxes.shape[0]):
                            fg_polys_ind = fg_polys_inds[j]
                            poly_gt = polys_gt[fg_polys_ind]
                            roi_fg = batch_mask_boxes[j]
                            # Rasterize the portion of the polygon mask within the given fg roi
                            # to an M x M binary image
                            assert batch_mask_boxes.shape[1] == 4
                            mask = segm_utils.polys_to_mask_wrt_box(poly_gt, roi_fg, M)
                            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
                            masks[count, :] = np.reshape(mask, M**2)
                            count += 1
                
                assert count == len(fg_inds)

            else:  # If there are no fg masks (it does happen)
                # The network cannot handle empty blobs, so we must provide a mask
                # We simply take the first bg roi, given it an all -1's mask (ignore
                # label), and label it with class zero (bg).
                bg_inds = np.where(roi_cls_inds == 0)[0]
                # rois_fg is actually one background roi, but that's ok because ...
                mask_rois = rois[bg_inds[0]].reshape((1, -1))
                # We give it an -1's blob (ignore label)
                masks = -blob_utils.ones((1, M**2), int32=True)
                # We label it with class = 0 (background)
                mask_class_labels = blob_utils.zeros((1, ))
                # Mark that the first roi has a mask
                #roi_has_mask[0] = 1
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

            
            blob_utils.py_op_copy_blob(mask_rois, outputs[0])
            blob_utils.py_op_copy_blob(masks, outputs[1])

        else:#testing time
            im_info = inputs[-1].data
            fg_inds = np.where(roi_cls_inds>0)[0]

            #output blob for testing
            mask_rois = None

            if len(fg_inds) > 0:
                fg_rois = rois[fg_inds]
                rois_deltas = np.empty((0,4),dtype=bbox_pred_odai.dtype)
                for k in range(len(bbox_pred_odai)):
                    rois_deltas = np.vstack((rois_deltas, bbox_pred_odai[k, 4*roi_cls_inds[k]:4*(roi_cls_inds[k]+1)]))
                fg_rois_deltas = rois_deltas[fg_inds, :]
                for i in range(im_info.shape[0]):
                    im_scale = im_info[i,-1]
                    batch_rois = fg_rois[np.where(fg_rois[:,0] == i)[0]]
                    batch_boxes = batch_rois[:,1:] / im_scale
                    batch_rois_deltas = fg_rois_deltas[np.where(fg_rois[:,0] == i)[0]]
                    # Transform fg_rois into proposals via bbox transformations
                    batch_mask_boxes = box_utils.bbox_transform(
                        batch_boxes, batch_rois_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
                    batch_mask_rois = batch_mask_boxes * im_scale
                    batch_mask_rois = np.hstack((batch_rois[:,0].reshape((-1,1)),batch_mask_rois))
                    if mask_rois == None:
                        mask_rois = np.copy(batch_mask_rois)
                    else:
                        mask_rois = np.vstack((mask_rois,batch_mask_rois))
            else:
                bg_inds = np.where(roi_cls_inds == 0)[0]
                # rois_fg is actually one background roi, but that's ok because ...
                mask_rois = rois[bg_inds[0]].reshape((1, -1))

            blob_utils.py_op_copy_blob(mask_rois, outputs[0])

def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
    to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    M = cfg.MRCNN.RESOLUTION

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -blob_utils.ones(
        (masks.shape[0], cfg.MODEL.NUM_CLASSES * M**2), int32=True
    )

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = M**2 * cls
        end = start + M**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets
