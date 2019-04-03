from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import logging

from collections import defaultdict

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import cv2
import tqdm

import torch
import pickle

import mask_rcnn.nn as mynn
from mask_rcnn.core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from mask_rcnn.core.test import im_just_det_cls
from model.mask_actor_action_det import FasterRCNNA2D
from mask_rcnn.utils.timer import Timer
from mask_rcnn.utils.detectron_weight_helper import load_detectron_weight

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=False)

    parser.add_argument('--test_lst', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--flow_root', type=str)
    parser.add_argument('--anno_root', type=str, required=True)
    parser.add_argument('--id_map_file', type=str, required=True)
    parser.add_argument('--det_result_pkl', type=str, required=True)
    parser.add_argument('--segment_length', type=int, default=4)

    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    id_to_actor_id_action_id_map = {}

    with open(args.id_map_file) as f:
        # each line is in such order: id, actor_id, action_id, actor_class, action_class
        lines = f.readlines()
        for line in lines:
            id, actor_id, action_id, actor_class, action_class = line.split(',')
            id, actor_id, action_id = int(id), int(actor_id), int(action_id)
            id_to_actor_id_action_id_map[id] = actor_id, action_id, actor_class, action_class

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.A2D.SEGMENT_LENGTH = args.segment_length
    assert cfg.A2D.SEGMENT_LENGTH % 2 == 0, 'SEGMENT_LENGTH must be even'
    print("segment length: {}".format(cfg.A2D.SEGMENT_LENGTH))

    cfg.A2D.ROOT = args.root
    cfg.A2D.FLOW_ROOT = args.flow_root
    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    fasterRCNNA2D = FasterRCNNA2D()

    if args.cuda:
        fasterRCNNA2D.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        # checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        # net_utils.load_ckpt(fasterRCNNA2D, checkpoint['model'])
        checkpoint = torch.load(load_name)
        fasterRCNNA2D.load_state_dict(checkpoint['model'], strict=True)
    elif args.load_detectron:
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(fasterRCNNA2D, args.load_detectron)

    # if args.load_detectron:
    #     print("loading detectron weights %s" % args.load_detectron)
    #     load_detectron_weight(fasterRCNNA2D, args.load_detectron)

    fasterRCNNA2D = mynn.DataParallel(fasterRCNNA2D, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU
    fasterRCNNA2D.eval()
    with open(args.test_lst) as f:
        imglist = [line.strip() for line in f.readlines()]

    num_images = len(imglist)

    det_result = {'actor': [[] for _ in range(8)], 'action': [[] for _ in range(10)], 'actor_action': [[] for _ in range(81)]}

    for i in tqdm.trange(num_images):
        im = cv2.imread(os.path.join(args.root, imglist[i] + '.png'))
        assert im is not None, os.path.join(args.root, imglist[i] + '.png')

        timers = defaultdict(Timer)

        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))

        pred_box_results = im_just_det_cls(fasterRCNNA2D, im, imglist[i], timers=timers)
        for record in pred_box_results:
            pred_actor_id, pred_action_id = record[-2:]
            pred_actor_id = int(pred_actor_id)
            pred_action_id = int(pred_action_id)
            pred_actor_action_id = 10 * pred_actor_id + pred_action_id
            det_result['actor'][pred_actor_id].append({'bbox': record, 'img': imglist[i]})
            det_result['action'][pred_action_id].append({'bbox': record, 'img': imglist[i]})
            det_result['actor_action'][pred_actor_action_id].append({'bbox': record, 'img': imglist[i]})

    with open(args.det_result_pkl, 'wb') as f:
        pickle.dump(det_result, f)
        print('finish')

if __name__ == '__main__':
    main()
