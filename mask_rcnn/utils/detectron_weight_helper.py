"""Helper functions for loading pretrained weights from Detectron pickle files
"""

import pickle
import re
import torch


def load_detectron_weight(net, detectron_weight_file):
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping

    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    for p_name, p_tensor in params.items():
        d_name = name_mapping[p_name]
        if isinstance(d_name, str) and d_name not in set(['cls_score_w', 'cls_score_b', 'bbox_pred_w',
                                                          'bbox_pred_b', 'mask_fcn_logits_w', 'mask_fcn_logits_b']):  # maybe str, None or True
            p_tensor.copy_(torch.Tensor(src_blobs[d_name]))


def resnet_weights_name_pattern():
    pattern = re.compile(r"conv1_w|conv1_gn_[sb]|res_conv1_.+|res\d+_\d+_.+")
    return pattern


if __name__ == '__main__':
    """Testing"""
    from pprint import pprint
    import sys
    # sys.path.insert(0, '..')
    from mask_rcnn.modeling.model_builder import Generalized_RCNN
    from mask_rcnn.core.config import cfg, cfg_from_file

    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False

    cfg.MODEL.NUM_CLASSES = 81
    cfg_from_file('/home/cxu-serve/u1/zli82/Project/metric/Detectron.pytorch/model_configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml')
    net = Generalized_RCNN()

    # pprint(list(net.state_dict().keys()), width=1)

    mapping, orphans = net.detectron_weight_mapping
    state_dict = net.state_dict()

    for k in mapping.keys():
        assert k in state_dict, '%s' % k

    rest = set(state_dict.keys()) - set(mapping.keys())
    assert len(rest) == 0
