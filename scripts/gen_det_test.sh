#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH='./':$PYTHONPATH python eval/baseline_gen_det_result.py \
    --dataset A2D \
    --cfg model_cfgs/e2e_faster_rcnn_R-50-FPN_1x.yaml \
    --image_dir /home/cxu-serve/p1/zli82/dataset/A2D/pngs320H \
    --test_lst /home/cxu-serve/p1/zli82/dataset/A2D/list/small_a2d_for_249/test.txt \
    --root /home/cxu-serve/p1/zli82/dataset/A2D/pngs320H \
    --flow_root /home/cxu-serve/p1/zli82/dataset/A2D/flow \
    --anno_root /home/cxu-serve/p1/zli82/dataset/A2D/Annotations/mat \
    --id_map_file /home/cxu-serve/p1/zli82/dataset/A2D/list/actor_id_action_id.txt \
    --det_result_pkl /home/cxu-serve/p1/zli82/249_final_proj_det/gen_det/model_step4749_baseline_small_val.pkl \
    --segment_length 2 \
    --load_ckpt /home/cxu-serve/p1/zli82/249_final_proj_det/train_output/e2e_faster_rcnn_R-50-FPN_1x/Mar30-16-34-40_iris_step/ckpt/model_step4749.pth
