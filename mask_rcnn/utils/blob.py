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
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import cv2
import os
import mmcv
from mask_rcnn.utils.compute_flow import flow_to_flow_img

from mask_rcnn.core.config import cfg


def get_image_blob(im, target_scale, target_max_size):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    processed_im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, [target_scale], target_max_size
    )
    blob = im_list_to_blob(processed_im)
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32)


def get_seq_blob(im, target_scale, target_max_size, vid_frm_idx):
    assert cfg.A2D.ROOT is not None
    vid, str_frm_idx = vid_frm_idx.split('/')
    cur_frm_idx = int(str_frm_idx)

    frames = []
    shared_frm_scales = None
    for f_idx in range(cur_frm_idx - int(cfg.A2D.SEGMENT_LENGTH / 2),
                       cur_frm_idx + int(cfg.A2D.SEGMENT_LENGTH / 2)):
        frame_fpath = os.path.join(cfg.A2D.ROOT, vid, '%05d' % f_idx + '.png')
        assert os.path.exists(frame_fpath)
        cur_frame = cv2.imread(frame_fpath)
        assert cur_frame is not None, 'Failed to read image {}'.format(frame_fpath)
        processed_frm, frm_scales = prep_im_for_blob(cur_frame, cfg.PIXEL_MEANS, [target_scale], target_max_size)
        assert len(processed_frm) == 1
        frames.append(processed_frm[0])
        if shared_frm_scales is None:
            shared_frm_scales = frm_scales
        else:
            assert shared_frm_scales == frm_scales

    blob = seq_list_to_blob([frames])
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, shared_frm_scales))[np.newaxis, :]
    return blob, shared_frm_scales, im_info.astype(np.float32)


def get_flow_blob(target_scale, target_max_size, vid_frm_idx):
    vid, str_frm_idx = vid_frm_idx.split('/')
    cur_frm_idx = int(str_frm_idx)

    flows = []
    shared_flow_scales = None
    for f_idx in range(cur_frm_idx - int(cfg.A2D.SEGMENT_LENGTH / 2),
                       cur_frm_idx + int(cfg.A2D.SEGMENT_LENGTH / 2)):
        flow_path = os.path.join(cfg.A2D.FLOW_ROOT, vid, '%05d' % f_idx + '.flo')
        assert os.path.exists(flow_path)
        flow = mmcv.flowread(flow_path)
        flows_with_diff_scales, computed_flow_scales = prep_flow_for_blob(
            flow, cfg.A2D.FLOW_MAX_MAG, [target_scale], target_max_size, clip_mag=cfg.A2D.CLIP_FLOW_MAG)
        three_channel_flow = flow_to_flow_img(flows_with_diff_scales[0])
        flows.append(three_channel_flow)
        if shared_flow_scales is not None:
            assert computed_flow_scales[0] == shared_flow_scales
        else:
            shared_flow_scales = computed_flow_scales[0]

    blob = flow_list_to_blob([flows])
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, shared_flow_scales))[np.newaxis, :]
    # return blob, shared_flow_scales, im_info.astype(np.float32)
    return blob

def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = get_max_shape([im.shape[:2] for im in ims])

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def seq_list_to_blob(seq_lst):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D BCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(seq_lst, list):
        seq_lst = [seq_lst]
    max_shape = get_max_shape([seq[0].shape[:2] for seq in seq_lst])

    num_segments = len(seq_lst)
    temporal_length = len(seq_lst[0])
    blob = np.zeros(
        (num_segments * temporal_length, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_segments):
        seq = seq_lst[i]
        assert isinstance(seq, list)
        seq = np.stack(seq) # [HWC] to BHWC
        blob[i:i+temporal_length, 0:seq.shape[1], 0:seq.shape[2], :] = seq
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def flow_list_to_blob(flow_segments):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D BCHW tensor of the images concatenated along axis 0 with
    shape.
    """

    # flow_segments is a list of flow_segment
    # flow_segment is a list of optical flow fields

    if not isinstance(flow_segments, list):
        flow_segments = [flow_segments]
    max_shape = get_max_shape([flow_segment[0].shape[:2] for flow_segment in flow_segments])

    num_flow_segments = len(flow_segments)
    temporal_length = len(flow_segments[0])

    # batch size = number of flow segments x length of flow segment
    blob = np.zeros(
        (num_flow_segments * temporal_length, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_flow_segments):
        flow_segment = flow_segments[i]  # flows is a list of optical flow segment
        assert isinstance(flow_segment, list)  # [HWC]
        flow_segment = np.stack(flow_segment)  # from [HWC] to BHWC
        blob[i:i+temporal_length, 0:flow_segment.shape[1], 0:flow_segment.shape[2], :] = flow_segment
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def get_max_shape(im_shapes):
    """Calculate max spatial size (h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
    return max_shape


def prep_im_for_blob(im, pixel_means, target_sizes, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        ims.append(im_resized)
        im_scales.append(im_scale)
    return ims, im_scales


def prep_flow_for_blob(flow, max_maginitude, target_sizes, max_size, clip_mag=False):
    """Prepare an optical flow for use as a network input blob. Specially:
          - clip the magnitude to range [-max_disp, max_disp]
          - Convert to float32
          - Rescale to each of the specified target size (capped at max_size)
        Returns a list of transformed flows, one for each target size. Also returns
        the scale factors that were used to compute each returned flow.
        """
    flow = flow.astype(np.float32, copy=False)
    if clip_mag:
        flow[flow > max_maginitude] = max_maginitude
        flow[flow < -max_maginitude] = -max_maginitude
    flow_shape = flow.shape
    flow_size_min = np.min(flow_shape[0:2])
    flow_size_max = np.max(flow_shape[0:2])

    flow_with_diff_scales = []
    flow_scales = []
    for target_size in target_sizes:
        flow_scale = get_target_scale(flow_size_min, flow_size_max, target_size, max_size)
        flow_resized = cv2.resize(flow, None, None, fx=flow_scale, fy=flow_scale,
                                  interpolation=cv2.INTER_LINEAR)
        # since the flow is resized, the magnitude also need to be resized
        flow_resized /= flow_scale
        flow_with_diff_scales.append(flow_resized)
        flow_scales.append(flow_scale)
    return flow_with_diff_scales, flow_scales


def get_im_blob_sizes(im_shape, target_sizes, max_size):
    """Calculate im blob size for multiple target_sizes given original im shape
    """
    im_size_min = np.min(im_shape)
    im_size_max = np.max(im_shape)
    im_sizes = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_sizes.append(np.round(im_shape * im_scale))
    return np.array(im_sizes)


def get_target_scale(im_size_min, im_size_max, target_size, max_size):
    """Calculate target resize scale
    """
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
