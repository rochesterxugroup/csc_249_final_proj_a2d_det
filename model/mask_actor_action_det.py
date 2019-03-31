import mask_rcnn.modeling.FPN as FPN
import mask_rcnn.modeling.rpn_heads as rpn_heads
import mask_rcnn.modeling.fast_rcnn_heads as fast_rcnn_heads
import mask_rcnn.modeling.mask_rcnn_heads as mask_rcnn_heads
import importlib
import logging

from mask_rcnn.core.config import cfg
import torch.nn as nn
import torch
from mask_rcnn.model.roi_align.functions.roi_align import RoIAlignFunction
import mask_rcnn.utils.resnet_weights_helper as resnet_utils
from mask_rcnn.modeling.model_builder import compare_state_dict
import mask_rcnn.utils.blob as blob_utils
import itertools
from functools import wraps


logger = logging.getLogger(__name__)

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'mask_rcnn.modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class FasterRCNNA2D(nn.Module):
    def __init__(self):
        super(FasterRCNNA2D, self).__init__()
        # FPN with ResNet101
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.RPN = rpn_heads.generic_rpn_outputs(
            self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Action_Box_Head = fast_rcnn_heads.roi_2mlp_head(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            # BBox branch following actor stream

            self.Actor_Box_Head = fast_rcnn_heads.roi_2mlp_head(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Actor_Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Actor_Box_Head.dim_out)
            self.Action_Box_Outs = fast_rcnn_heads.fast_rcnn_action_outputs(
                self.Action_Box_Head.dim_out
            )

        # Mask branch following actor stream
        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = mask_rcnn_heads.mask_rcnn_fcn_head_v1up4convs(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Actor_Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        self._init_modules()

    def forward(self, data, flow, im_info, roidb=None, **rpn_kwargs):
        b, _, c, h, w = data.size()
        segment_data = data.view(-1, c, h, w)

        with torch.set_grad_enabled(self.training):
            if self.training:
                roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

            return_dict = {}  # A dict to collect return variables

            actor_blob_conv = []
            segment_actor_blob_conv_pyramid = self.Conv_Body(segment_data)
            for segment_actor_blob_conv_level_feat in segment_actor_blob_conv_pyramid:
                indices = [(2 * k + 1) * int(cfg.A2D.SEGMENT_LENGTH / 2) for k in range(b)]
                frame_feature_level = torch.index_select(segment_actor_blob_conv_level_feat, dim=0, index=
                torch.tensor(indices).to(segment_actor_blob_conv_level_feat.device))
                actor_blob_conv.append(frame_feature_level)
                # for k in range(b):
                #      = segment_actor_blob_conv_level_feat[(2 * k + 1) * int(cfg.A2D.SEGMENT_LENGTH / 2)]
                #     frame_feature_level.append(frame_feature_level)

            rpn_ret = self.RPN(actor_blob_conv, im_info, roidb)

            # if self.training:
            #     # can be used to infer fg/bg ratio
            #     return_dict['rois_label'] = rpn_ret['labels_int32']

            if cfg.FPN.FPN_ON:
                # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
                # extra blobs that are used for RPN proposals, but not for RoI heads.
                actor_blob_conv = actor_blob_conv[-self.num_roi_levels:]
                # segment_actor_blob_conv_pyramid = segment_actor_blob_conv_pyramid[-self.num_roi_levels:]

            if not self.training:
                return_dict['blob_conv'] = actor_blob_conv

            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Actor_Box_Head(actor_blob_conv, rpn_ret)
            else:
                box_feat = self.Actor_Box_Head(actor_blob_conv, rpn_ret)
            actor_cls_score, actor_bbox_pred = self.Actor_Box_Outs(box_feat)

            action_box_feat = self.Action_Box_Head(actor_blob_conv, rpn_ret)
            action_cls_score = self.Action_Box_Outs(action_box_feat)

            if self.training:
                return_dict['losses'] = {}
                return_dict['metrics'] = {}

                # rpn loss
                rpn_kwargs.update(dict(
                    (k, rpn_ret[k]) for k in rpn_ret.keys()
                    if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
                ))
                loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
                if cfg.FPN.FPN_ON:
                    for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                        return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                        return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
                else:
                    return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                    return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

                # bbox loss
                loss_actor_cls, loss_bbox, actor_accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                    actor_cls_score, actor_bbox_pred, rpn_ret['labels_int32'], rpn_ret['actor_bbox_targets'],
                    rpn_ret['actor_bbox_inside_weights'], rpn_ret['actor_bbox_outside_weights'])
                loss_action_cls, action_accuracy_cls = fast_rcnn_heads.fast_rcnn_action_losses(action_cls_score, rpn_ret['action_labels_int32'])

                return_dict['losses']['loss_actor_cls'] = loss_actor_cls
                return_dict['losses']['loss_action_cls'] = loss_action_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['actor_accuracy_cls'] = actor_accuracy_cls
                return_dict['metrics']['action_accuracy_cls'] = action_accuracy_cls

                if cfg.MODEL.MASK_ON:
                    if getattr(self.Mask_Head, 'SHARE_RES5', False):
                        mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                                   roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                    else:
                        mask_feat = self.Mask_Head(actor_blob_conv, rpn_ret)
                    mask_pred = self.Mask_Outs(mask_feat)
                    return_dict['mask_pred'] = mask_pred
                    # mask loss
                    loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                    return_dict['losses']['loss_mask'] = loss_mask

                # pytorch0.4 bug on gathering scalar(0-dim) tensors
                for k, v in return_dict['losses'].items():
                    return_dict['losses'][k] = v.unsqueeze(0)
                for k, v in return_dict['metrics'].items():
                    return_dict['metrics'][k] = v.unsqueeze(0)

            else:
                # Testing
                return_dict['rois'] = rpn_ret['rois']
                return_dict['actor_cls_score'] = actor_cls_score
                return_dict['action_cls_score'] = action_cls_score
                return_dict['bbox_pred'] = actor_bbox_pred

            return return_dict

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Actor_Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in itertools.chain(self.Conv_Body.parameters()):
                p.requires_grad = False

    @staticmethod
    def roi_feature_transform(blobs_in, rpn_ret, blob_rois='rois', method='RoIAlign',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = torch.from_numpy(rpn_ret[bl_rois]).cuda(device_id)
                    # for RoIAlign
                    # xform_out = RoIAlignFunction(
                    #     resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    xform_out = RoIAlignFunction(
                        resolution, resolution, sc)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = torch.from_numpy(restore_bl.astype('int64', copy=False)).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = torch.from_numpy(rpn_ret[blob_rois]).cuda(device_id)
            xform_out = RoIAlignFunction(
                resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @property
    def detectron_weight_mapping(self):
        d_wmap = {}  # detectron_weight_mapping
        d_orphan = []  # detectron orphan weight list
        for name, m_child in self.named_children():
            if list(m_child.parameters()):  # if module has any parameter
                child_map, child_orphan = m_child.detectron_weight_mapping()
                d_orphan.extend(child_orphan)
                for key, value in child_map.items():
                    new_key = name + '.' + key
                    d_wmap[new_key] = value
        self.mapping_to_detectron = d_wmap
        self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron
