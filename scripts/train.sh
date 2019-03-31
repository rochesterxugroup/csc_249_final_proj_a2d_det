#!/usr/bin/env bash
export DATA_SET_ROOT='../A2D'
PYTHONPATH='./':$PYTHONPATH python train.py \
    --dataset A2D \
    --cfg model_cfgs/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    --bs 3 \
    --nw 0 \
    --lr 2e-4 \
    --train_lst $DATA_SET_ROOT/list/train.txt \
    --annotation_root $DATA_SET_ROOT/Annotations/new_mat \
    --frame_root $DATA_SET_ROOT/pngs320H \
    --id_map_file $DATA_SET_ROOT/list/actor_id_action_id.txt \
    --dataset A2D \
    --use_tfboard \
    --segment_length 2 \
    --output_dir ./train_output \
    --snapshot_iters 4750
