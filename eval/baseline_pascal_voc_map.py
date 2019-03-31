import numpy as np
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_root', type=str)
    parser.add_argument('--mode', type=str, default='actor')
    parser.add_argument('--gt_cls_pkl', type=str, required=True)
    parser.add_argument('--det_cls_pkl', type=str, required=True)
    args = parser.parse_args()

    if args.mode == 'actor_action':
        classes = [11, 12, 13, 15, 16, 17, 18, 19,    # 1-8
                    21, 22, 26, 28, 29,    # 9-13
                    34, 35, 36, 39,    # 14-17
                    41, 43, 44, 45, 46, 48, 49,    # 18-24
                    54, 55, 56, 57, 59,    # 25-29
                    61, 63, 65, 66, 67, 68, 69,    # 30-36
                    72, 73, 75, 76, 77, 78, 79] # 37-43
    elif args.mode == 'actor':
        classes = range(1, 8)
    elif args.mode == 'action':
        classes = range(1, 10)

    assert classes is not None

    with open(args.gt_cls_pkl, 'rb') as gt_f, open(args.det_cls_pkl, 'rb') as det_f:
        gt_cls_det = pickle.load(gt_f)[args.mode]
        pred_cls_det = pickle.load(det_f)[args.mode]

        gt_cls_det_lst = []
        for records in gt_cls_det:
            gt_cls_det_dict = {}
            for record in records:
                if record['img'] not in gt_cls_det_dict:
                    gt_cls_det_dict[record['img']] = []
                gt_cls_det_dict[record['img']].append(record['box'])
            gt_cls_det_lst.append(gt_cls_det_dict)

        pred_cls_det_lst = []
        for records in pred_cls_det:
            pred_cls_det_dict = {}
            for record in records:
                if record['img'] not in pred_cls_det_dict:
                    pred_cls_det_dict[record['img']] = []
                pred_cls_det_dict[record['img']].append(record['bbox'])
            pred_cls_det_lst.append(pred_cls_det_dict)

    aps = []
    recalls = []
    precisions = []
    for class_id in classes:
        recall, precision, ap = voc_eval_of_one_class(class_id,
                                                      gt_cls_det_lst[class_id],
                                                      pred_cls_det_lst[class_id],
                                                      args.mode)
        if len(pred_cls_det_lst[class_id]) > 0:
            print('class {}, max recall: {}'.format(class_id, recall[-1]))
            print('class {}, overall precision: {}'.format(class_id, precision[-1]))

        if len(pred_cls_det_lst[class_id]) > 0:
            recalls.append(recall[-1])
            precisions.append(precision[-1])
        else:
            recalls.append(0)
            precisions.append(0)
        print('class {}, ap: {}'.format(class_id, ap))
        aps.append(ap)

    print('mode: {}'.format(args.mode))
    for class_id, recall, precision, ap in zip(classes, recalls, precisions, aps):
        print('class_id: {}, max recall: {}, precision: {}  ap: {}'.format(class_id, recall, precision, ap))
    # print('mean max recall: {}'.format(np.mean(recalls)))
    # print('mean precision: {}'.format(np.mean(precisions)))
    print('mean ap: {}'.format(np.mean(aps)))


def voc_eval_of_one_class(class_id,
                          gt_det_dict,
                          pred_det_dict,
                          mode,
                          ovthresh=0.5,
                          use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # read list of images
    npos = 0

    class_recs = {}
    for vid_frm_idx, boxes in gt_det_dict.items():
        npos += len(boxes)
        class_recs[vid_frm_idx] = {'bbox': np.array(boxes), 'det': [False] * len(boxes)}

    image_ids = []
    BB = np.array([]).reshape(0, 4)

    print('load detections of class: {}'.format(class_id))

    confidence = []
    for pred_vid_frm_idx, records in pred_det_dict.items():
        for record in records:
            bbox = record[:4].reshape(1, 4)
            conf = 0
            if mode == 'actor_action':
                conf = float(record[4]) * float(record[5])
            elif mode == 'actor':
                conf = float(record[4])
            elif mode == 'action':
                conf = float(record[5])
            image_ids.append(pred_vid_frm_idx)
            confidence.append(conf)
            BB = np.concatenate([BB, bbox], axis=0)

    assert len(image_ids) == BB.shape[0]
    assert len(image_ids) == len(confidence)

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]
    image_ids = np.array(image_ids)[sorted_ind].tolist()

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in class_recs:
            R = {'bbox': np.array([]), 'det': []}
        else:
            R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=True):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    main()