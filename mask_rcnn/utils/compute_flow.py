import numpy as np


def flow_to_flow_img(flow):
    """
    convert a two channel into flow_image
    MATLAB implementation: https://github.com/gkioxari/ActionTubes/blob/master/compute_OF/compute_flow.m
    :param flow:
    :return:
    """
    assert flow.shape[2] == 2
    max_flow = 8
    scale = 128 / max_flow
    mag_flow = np.sqrt(np.sum(flow ** 2, axis=2))
    flow = flow * scale + 128
    flow[flow < 0] = 0
    flow[flow > 255] = 255

    mag_flow = mag_flow * scale + 128
    mag_flow[mag_flow < 0] = 0
    mag_flow[mag_flow > 255] = 255

    flow_img = np.concatenate([flow, np.expand_dims(mag_flow, axis=2)], axis=2).astype(np.uint8)
    return flow_img
