import os
import hdf5storage
import cv2
import numpy as np
import random
import scipy.sparse
from mask_rcnn.core.config import cfg
import logging
import math
import pickle
import tqdm
import multiprocessing

global_annotation_root = None
global_id_to_actor_id_action_id_map = None
global_frame_root = None


def _worker(line):
    global global_annotation_root
    global global_id_to_actor_id_action_id_map
    global global_frame_root

    mat_path = os.path.join(global_annotation_root, line[:-1] + '.mat')

    assert os.path.isfile(mat_path), mat_path

    mat = hdf5storage.loadmat(mat_path)
    bboxes = mat[u'reBBox']
    ids = mat[u'id'].flatten().astype(np.int32)

    item = {}
    item['image'] = line[:-1]
    height, width, _ = cv2.imread(os.path.join(global_frame_root, item['image']) + '.png').shape
    item['height'] = height
    item['width'] = width
    item['flipped'] = False
    item['has_visible_keypoints'] = False
    item['is_crowd'] = np.array([False] * len(bboxes))
    item['box_to_gt_ind_map'] = np.arange(len(bboxes), dtype=np.int32)
    item['actor_max_overlaps'] = np.ones(shape=range(len(bboxes)))  # prob is float
    item['action_max_overlaps'] = np.ones(shape=range(len(bboxes)))  # prob is float

    # borrow the code from rank_for_training
    ratio = width / float(height)
    if cfg.TRAIN.ASPECT_CROPPING:
        if ratio > cfg.TRAIN.ASPECT_HI:
            item['need_crop'] = True
            ratio = cfg.TRAIN.ASPECT_HI
            # need_crop_cnt += 1
        elif ratio < cfg.TRAIN.ASPECT_LO:
            item['need_crop'] = True
            ratio = cfg.TRAIN.ASPECT_LO
            # need_crop_cnt += 1
        else:
            item['need_crop'] = False
    else:
        item['need_crop'] = False
    # end of rank_for_training

    boxes = []
    actor_classes = []  # as gt_classes
    action_classes = []  # as gt_classes
    actor_bbox_targets = []
    action_bbox_targets = []
    actor_gt_overlap = np.zeros(shape=(len(bboxes), 8))  # including background
    action_gt_overlap = np.zeros(shape=(len(bboxes), 10))
    seg_areas = np.zeros(len(bboxes))

    for enm_idx, (bbox, id) in enumerate(zip(bboxes, ids)):
        actor_id, action_id, _, _ = global_id_to_actor_id_action_id_map[id]
        boxes.append(bbox)
        actor_classes.append(actor_id)
        action_classes.append(action_id)
        actor_bbox_targets.append([actor_id] + bbox)
        action_bbox_targets.append([action_id] + bbox)
        actor_gt_overlap[enm_idx][actor_id] = 1.0
        action_gt_overlap[enm_idx][action_id] = 1.0

        # calculate seg_area:
        # for each box, count number of elements of the id within that box rectangular
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = int(math.floor(x_min)), int(math.floor(y_min))
        x_max, y_max = min(int(math.ceil(x_max)), width), min(int(math.ceil(y_max)), height)
        crop_seg = mat[u'reS_id'][y_min: y_max, x_min: x_max]
        seg_areas[enm_idx] = np.count_nonzero(crop_seg == id)

    item['boxes'] = np.array(boxes, dtype=np.float32).reshape((len(bboxes), 4))
    item['actor_gt_classes'] = np.array(actor_classes)
    item['action_gt_classes'] = np.array(action_classes)
    item['actor_max_classes'] = np.array(actor_classes)
    item['action_max_classes'] = np.array(action_classes)
    # item['actor_bbox_targets'] = np.array(actor_bbox_targets)
    # item['action_bbox_targets'] = np.array(action_bbox_targets)
    item['actor_gt_overlaps'] = scipy.sparse.csr_matrix(actor_gt_overlap)
    item['action_gt_overlaps'] = scipy.sparse.csr_matrix(action_gt_overlap)
    item['seg_areas'] = np.array(seg_areas)

    return item, ratio


def load_A2D_from_list_in_COCO_format(lst_fpath, annotation_root, id_map_file, frame_root):
    global global_annotation_root
    global global_id_to_actor_id_action_id_map
    global global_frame_root

    id_to_actor_id_action_id_map = {}

    with open(id_map_file) as f:
        # each line is in such order: id, actor_id, action_id, actor_class, action_class
        lines = f.readlines()
        for line in lines:
            #
            id, actor_id, action_id, actor_class, action_class = line.split(',')
            # id + 1 to add background class
            id, actor_id, action_id = int(id), int(actor_id) + 1, int(action_id) + 1
            id_to_actor_id_action_id_map[id] = actor_id, action_id, actor_class, action_class

    global_id_to_actor_id_action_id_map = id_to_actor_id_action_id_map
    global_annotation_root = annotation_root
    global_frame_root = frame_root

    result = []
    # need_crop_cnt = 0
    ratio_list = []

    with open(lst_fpath, 'r') as f:
        lines = f.readlines()
        # only load 20 items for debug (needs to be removed when release)
        if cfg.DEBUG:
            lines = random.sample(lines, 50)

        if cfg.DATA_LOADER.NUM_THREADS == 1 or cfg.DATA_LOADER.NUM_THREADS == 0:
            for line in tqdm.tqdm(lines):
                item, ratio = _worker(line)
                result.append(item)
                ratio_list.append(ratio)
        else:
            pool = multiprocessing.Pool(cfg.DATA_LOADER.NUM_THREADS)
            for item, ratio in tqdm.tqdm(pool.imap_unordered(_worker, lines), total=len(lines)):
                result.append(item)
                ratio_list.append(ratio)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Ratio bound: [%.2f, %.2f]',
                     cfg.TRAIN.ASPECT_LO, cfg.TRAIN.ASPECT_HI)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)

    return result, ratio_list[ratio_index], ratio_index
