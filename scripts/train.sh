#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset A2D \
    --cfg model_cfgs/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    --bs 3 \
    --nw 0 \
    --lr 2e-4 \
    --train_lst /home/cxu-serve/p1/zli82/dataset/A2D/list/small_a2d_for_249/train.txt \
    --annotation_root /home/cxu-serve/p1/zli82/dataset/A2D/Annotations/new_mat \
    --frame_root /home/cxu-serve/p1/zli82/dataset/A2D/pngs320H \
    --id_map_file /home/cxu-serve/p1/zli82/dataset/A2D/list/actor_id_action_id.txt \
    --dataset A2D \
    --use_tfboard \
    --segment_length 2 \
    --load_detectron /home/cxu-serve/p1/zli82/249_final_proj_det/pretrained_weights/model_final.pkl \
    --output_dir /home/cxu-serve/p1/zli82/249_final_proj_det/train_output \
    --snapshot_iters 4750
