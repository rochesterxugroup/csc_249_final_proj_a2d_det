#!/usr/bin/env bash
export DATA_SET_ROOT='../A2D'
PYTHONPATH='./':$PYTHONPATH python eval/baseline_gen_det_result.py \
    --dataset A2D \
    --cfg model_cfgs/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    --image_dir $DATA_SET_ROOT/pngs320H \
    --test_lst $DATA_SET_ROOT/list/small_a2d_for_249/val.txt \
    --root $DATA_SET_ROOT/pngs320H \
    --flow_root $DATA_SET_ROOT/flow \
    --anno_root $DATA_SET_ROOT/Annotations/mat \
    --id_map_file $DATA_SET_ROOT/list/actor_id_action_id.txt \
    --det_result_pkl /home/cxu-serve/p1/zli82/249_final_proj_det/gen_det/model_step4749_baseline_small_val.pkl \
    --segment_length 2 \
    --load_ckpt /home/cxu-serve/p1/zli82/249_final_proj_det/train_output/e2e_faster_rcnn_R-50-FPN_1x/Mar30-16-34-40_iris_step/ckpt/model_step4749.pth
