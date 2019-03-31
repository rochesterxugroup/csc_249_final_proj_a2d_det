import numpy as np
import cv2

from mask_rcnn.core.config import cfg
import mask_rcnn.utils.blob as blob_utils
import mask_rcnn.roi_data.rpn
from mask_rcnn import roi_data
import mmcv
import os
from mask_rcnn.utils.compute_flow import flow_to_flow_img


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb, frame_root, flow_root):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    # im_blob, im_scales = _get_image_blob(roidb, frame_root)
    seq_blob, seq_scales = _get_image_seq_blob(roidb, frame_root)
    # get the flow blob
    flow_blob, flow_scales = _get_flow_blob(roidb, frame_root, flow_root)

    blobs['data'] = seq_blob
    blobs['flow'] = flow_blob
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, seq_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, seq_scales, roidb)
    return blobs, valid


def _get_image_blob(roidb, frame_root):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(os.path.join(frame_root, roidb[i]['image'] + '.png'))
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_image_seq_blob(roidb, frame_root):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_segments = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_segments)
    processed_seqs = []
    seq_scales = []
    for i in range(num_segments):
        video_name, str_frame_index = roidb[i]['image'].split('/')
        frame_index = int(str_frame_index)

        segment = []
        cur_seg_scale = None

        for frm_idx in range(frame_index - int(cfg.A2D.SEGMENT_LENGTH/2), frame_index + int(cfg.A2D.SEGMENT_LENGTH/2)):
            frame_fpath = os.path.join(frame_root, video_name, '%05d' % frm_idx + '.png')
            assert os.path.exists(frame_fpath)
            cur_frame = cv2.imread(frame_fpath)
            assert cur_frame is not None, 'Failed to read image {}'.format(frame_fpath)

            if roidb[i]['flipped']:
                cur_frame = cur_frame[:, ::-1, :]

            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            cur_frames, frm_scales = blob_utils.prep_im_for_blob(
                cur_frame, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
            if cur_seg_scale is not None:
                assert cur_seg_scale == frm_scales[0]
            else:
                cur_seg_scale = frm_scales[0]
            segment.append(cur_frames[0])

        seq_scales.append(cur_seg_scale)
        processed_seqs.append(segment)

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.seq_list_to_blob(processed_seqs)

    return blob, seq_scales


def _get_flow_blob(roidb, frame_root, flow_root):
    num_images = len(roidb)
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_flows = []
    final_flow_scales = []
    for i in range(num_images):
        video_name, str_frame_index = roidb[i]['image'].split('/')
        frame_index = int(str_frame_index)

        flows = []
        flow_scales = None

        if cfg.A2D.LOAD_FLOW:
            if flow_root is None or not os.path.exists(flow_root):
                raise RuntimeError('flow root not provided or flow root not exists')
            for f_idx in range(frame_index - int(cfg.A2D.SEGMENT_LENGTH / 2),
                               frame_index + int(cfg.A2D.SEGMENT_LENGTH / 2)):
                flow_path = os.path.join(flow_root, video_name, '%05d' % f_idx + '.flo')
                assert os.path.exists(flow_path)
                flow = mmcv.flowread(flow_path)
                if roidb[i]['flipped']:
                    flow = flow[:, ::-1, :]
                target_size = cfg.TRAIN.SCALES[scale_inds[i]]
                flows_with_diff_scales, computed_flow_scales = blob_utils.prep_flow_for_blob(
                    flow, cfg.A2D.FLOW_MAX_MAG, [target_size], cfg.TRAIN.MAX_SIZE, clip_mag=cfg.A2D.CLIP_FLOW_MAG)
                three_channel_flow = flow_to_flow_img(flows_with_diff_scales[0])
                flows.append(three_channel_flow)
                if flow_scales is not None:
                    assert computed_flow_scales[0] == flow_scales
                else:
                    flow_scales = computed_flow_scales[0]
        else:
            prev_frame = None
            for f_idx in range(frame_index - int(cfg.A2D.SEGMENT_LENGTH/2),
                               frame_index + int(cfg.A2D.SEGMENT_LENGTH/2) + 1):
                frame_fpath = os.path.join(frame_root, video_name, '%05d' % f_idx + '.png')
                assert os.path.exists(frame_fpath)
                cur_frame = cv2.imread(frame_fpath)
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    if roidb[i]['flipped']:
                        flow = flow[:, ::-1, :]
                    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
                    flows_with_diff_scales, computed_flow_scales = blob_utils.prep_flow_for_blob(
                        flow, cfg.A2D.FLOW_MAX_MAG, [target_size], cfg.TRAIN.MAX_SIZE, clip_mag=cfg.A2D.CLIP_FLOW_MAG)
                    three_channel_flow = flow_to_flow_img(flows_with_diff_scales[0])
                    flows.append(three_channel_flow)
                    if flow_scales is not None:
                        assert computed_flow_scales[0] == flow_scales
                    else:
                        flow_scales = computed_flow_scales[0]

                prev_frame = cur_frame
        processed_flows.append(flows)
        final_flow_scales.append(flow_scales)

    blob = blob_utils.flow_list_to_blob(processed_flows)
    return blob, final_flow_scales
