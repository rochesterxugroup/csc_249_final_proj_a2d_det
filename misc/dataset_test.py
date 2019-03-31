from mask_rcnn.roi_data.loader import RoiDataLoader, MinibatchSampler, \
    BatchSampler, collate_minibatch
from dataset.A2DCOCO import load_A2D_from_list_in_COCO_format
from mask_rcnn.core.config import cfg, cfg_from_file
from torch.utils.data import DataLoader
from model.mask_actor_action_det import FasterRCNNA2D
from mask_rcnn.utils.detectron_weight_helper import load_detectron_weight

cfg_from_file('model_cfgs/e2e_faster_rcnn_R-50-FPN_1x.yaml')

cfg.DEBUG = True
cfg.DATA_LOADER.NUM_THREADS = 32

# roidb, ratio_list, ratio_index = load_A2D_from_list_in_COCO_format(
#     lst_fpath='/home/cxu-serve/p1/zli82/dataset/A2D/list/train_annotated_neighbor_4_poly.txt',
#     annotation_root='/home/cxu-serve/p1/zli82/dataset/A2D/Annotations/new_mat',
#     id_map_file='/home/cxu-serve/p1/zli82/dataset/A2D/list/actor_id_action_id.txt',
#     frame_root='/home/cxu-serve/p1/zli82/dataset/A2D/pngs320H',
# )
#
# dataset = RoiDataLoader(roidb,
#                         frame_root='/home/cxu-serve/p1/zli82/dataset/A2D/pngs320H',
#                         flow_root='/home/cxu-serve/p1/zli82/dataset/A2D/flow')
#
# batchSampler = BatchSampler(
#         sampler=MinibatchSampler(ratio_list, ratio_index),
#         batch_size=1,
#         drop_last=True
# )
#
# dataloader = DataLoader(
#         dataset,
#         batch_sampler=batchSampler,
#         num_workers=0,
#         collate_fn=collate_minibatch)
cfg_from_file('model_cfgs/e2e_faster_rcnn_R-50-FPN_1x.yaml')
cfg.TRAIN.DATASETS = ('coco_2017_train',)
cfg.MODEL.NUM_CLASSES = 81
cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
cfg.MODEL.NUM_ACTOR_CLASSES = 8
cfg.MODEL.NUM_ACTION_CLASSES = 10
model = FasterRCNNA2D()
load_detectron_weight(model, '/home/cxu-serve/p1/zli82/249_final_proj_det/pretrained_weights/model_final.pkl')
