import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import mask_rcnn.nn as mynn
import mask_rcnn.utils.net as net_utils
import mask_rcnn.utils.misc as misc_utils
from mask_rcnn.core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from mask_rcnn.roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from mask_rcnn.utils.logging import setup_logging
from mask_rcnn.utils.detectron_weight_helper import load_detectron_weight
from mask_rcnn.utils.training_stats import TrainingStats
from dataset.A2DCOCO import load_A2D_from_list_in_COCO_format
from model.mask_actor_action_det import FasterRCNNA2D

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Finetune MaskA2D on A2D dataset')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')
    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=20, type=int)

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw',
        dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 1',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--resume_ckpt', help='checkpoint path to load to resume training')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard',
        help='Use tensorflow tensorboard to log training info',
        action='store_true')

    parser.add_argument('--train_lst', type=str, required=True,
                        help='path to the txt file containing filenames of training frames in A2D dataset')
    parser.add_argument('--annotation_root', type=str, required=True, help='directory of annotation')
    parser.add_argument('--id_map_file', type=str, required=True,
                        help='path to a txt file containing the map between actor_action id to actor id and action id')
    parser.add_argument('--frame_root', type=str, required=True, help='directory of video frames')
    # parser.add_argument('--flow_root', type=str, required=True, help='directory of ')
    parser.add_argument('--segment_length', type=int, default=2)
    parser.add_argument('--debug', action='store_true',
                        help='turn on debug mode. If it\'s on: '
                             '1. checkpoint of model will not be saved\n'
                             '2. tensorboard will not be used\n'
                             '3. training set will sampled to 50 elements for faster data loading')
    parser.add_argument('--output_dir', type=str,
                        help='directory of outputs, which will contain '
                             'tensorboard records, checkpoints of model, args of the training.')
    parser.add_argument('--snapshot_iters', type=int, default=20000,
                        help='interval of saving a checkpoint of the model')

    args = parser.parse_args()
    if args.resume:
        assert args.resume_ckpt is not None

    return args


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.debug:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not args.debug:
        assert args.output_dir is not None
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        cfg.OUTPUT_DIR = args.output_dir

    assert args.dataset == 'A2D'

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.A2D.LOAD_FLOW = False
    cfg.TRAIN.SNAPSHOT_ITERS = args.snapshot_iters
    cfg.MODEL.NUM_CLASSES = -1
    cfg.MODEL.NUM_ACTOR_CLASSES = 8
    cfg.MODEL.NUM_ACTION_CLASSES = 10
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
    cfg.A2D.SEGMENT_LENGTH = args.segment_length
    cfg.DEBUG = args.debug

    assert cfg.A2D.SEGMENT_LENGTH % 2 == 0, 'SEGMENT_LENGTH must be even'
    # assert cfg.TEST.SEGMENT_LENGTH % 2 == 0, 'SEGMENT_LENGTH must be even'

    ### Adaptively adjust some model_configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    effective_batch_size = args.iter_size * args.batch_size
    print(
        'effective_batch_size = batch_size * iter_size = %d * %d' %
        (args.batch_size, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' %
          (original_batch_size, effective_batch_size))
    print('    NUM_GPUS:             %d --> %d' %
          (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' %
          (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

    # Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

    # Adjust solver steps
    step_scale = original_batch_size / effective_batch_size
    old_solver_steps = cfg.SOLVER.STEPS
    old_max_iter = cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = list(
        map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
    print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                                  old_max_iter, cfg.SOLVER.MAX_ITER))

    # Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
    # of `collect_and_distribute_fpn_rpn_proposals.py`
    #
    # post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print(
            'Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
            '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(
                cfg.FPN.RPN_COLLECT_SCALE))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    # Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma
    assert_and_infer_cfg()

    ### Dataset ###
    # timers['roidb'].tic()
    # roidb, ratio_list, ratio_index = combined_roidb_for_training(
    #     cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    # timers['roidb'].toc()
    roidb, ratio_list, ratio_index = load_A2D_from_list_in_COCO_format(args.train_lst,
                                                                       args.annotation_root,
                                                                       args.id_map_file,
                                                                       args.frame_root)
    roidb_size = len(roidb)
    # logger.info('{:d} roidb entries'.format(roidb_size))
    # logger.info(
    #     'Takes %.2f sec(s) to construct roidb',
    #     timers['roidb'].average_time)

    # Effective training sample size for one epoch
    train_size = roidb_size // args.batch_size * args.batch_size

    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=args.batch_size,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        args.frame_root,
        args.flow_root,
        training=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)

    ### Model ###
    fasterRCNNA2D = FasterRCNNA2D()

    if cfg.CUDA:
        fasterRCNNA2D.cuda()

    ### Optimizer ###
    gn_param_nameset = set()
    for name, module in fasterRCNNA2D.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_param_nameset.add(name + '.weight')
            gn_param_nameset.add(name + '.bias')
    gn_params = []
    gn_param_names = []
    bias_params = []
    bias_param_names = []
    nonbias_params = []
    nonbias_param_names = []
    nograd_param_names = []
    for key, value in fasterRCNNA2D.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
                bias_param_names.append(key)
            elif key in gn_param_nameset:
                gn_params.append(value)
                gn_param_names.append(key)
            else:
                nonbias_params.append(value)
                nonbias_param_names.append(key)
        else:
            nograd_param_names.append(key)
    assert (gn_param_nameset - set(nograd_param_names) -
            set(bias_param_names)) == set(gn_param_names)

    # Learning rate of 0 is a dummy value to be set properly at the start of
    # training
    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)
    else:
        print('invalid solver: {}'.format(cfg.SOLVER.TYPE))
        sys.exit(-1)

    if args.load_detectron:
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(fasterRCNNA2D, args.load_detectron)

    if args.resume:
        print('resume training')
        load_name = args.resume_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(
            load_name,
            map_location=lambda storage,
            loc: storage)
        net_utils.load_ckpt(fasterRCNNA2D, checkpoint['model'])
        args.start_step = checkpoint['step'] + 1
        if 'train_size' in checkpoint:  # For backward compatibility
            if checkpoint['train_size'] != train_size:
                print(
                    'train_size value: %d different from the one in checkpoint: %d' %
                    (train_size, checkpoint['train_size']))

        # reorder the params in optimizer checkpoint's params_groups if needed
        # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

        # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
        # However it's fixed on master.
        optimizer.load_state_dict(checkpoint['optimizer'])
        # misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
    # else:
    #     fasterRCNNA2D.load_state_dict(torch.load(args.ckpt), strict=False)

    # lr of non-bias parameters, for commmand line outputs.
    lr = optimizer.param_groups[0]['lr']

    fasterRCNNA2D = mynn.DataParallel(fasterRCNNA2D, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True).cuda()

    ### Training Setups ###
    if args.resume:
        args.run_name = misc_utils.get_run_name() + '_step'
        output_dir = misc_utils.get_output_dir(args, args.run_name)
        args.cfg_filename = os.path.basename(args.cfg_file)
    else:
        args.run_name = misc_utils.get_run_name() + '_step'
        output_dir = misc_utils.get_output_dir(args, args.run_name)
        args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.debug:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    ### Training Loop ###
    fasterRCNNA2D.train()

    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)

    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] >= args.start_step:
            decay_steps_ind = i
            break
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.debug else None)
    try:
        logger.info('Training starts !')
        step = args.start_step
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):
            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * \
                        (1 - alpha) + alpha
                else:
                    raise KeyError(
                        'Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate(
                    optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR

            # Learning rate decay
            if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                    step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                decay_steps_ind += 1

            training_stats.IterTic()
            optimizer.zero_grad()
            fasterRCNNA2D.zero_grad()
            for inner_iter in range(args.iter_size):
                try:
                    input_data = next(dataiterator)
                except StopIteration:
                    dataiterator = iter(dataloader)
                    input_data = next(dataiterator)

                for key in input_data:
                    if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))

                net_outputs = fasterRCNNA2D(**input_data)
                training_stats.UpdateIterStats(net_outputs, inner_iter)
                loss = net_outputs['total_loss']
                loss.backward()
            optimizer.step()
            training_stats.IterToc()

            training_stats.LogIterStats(step, lr)

            if (step + 1) % CHECKPOINT_PERIOD == 0:
                save_ckpt(
                    output_dir,
                    args,
                    step,
                    train_size,
                    fasterRCNNA2D,
                    optimizer)

        # ---- Training ends ----
        # Save last checkpoint
        save_ckpt(output_dir, args, step, train_size, fasterRCNNA2D, optimizer)

    except (RuntimeError, KeyboardInterrupt):
        del dataiterator
        logger.info('Save ckpt on exception ...')
        save_ckpt(output_dir, args, step, train_size, fasterRCNNA2D, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.debug:
            tblogger.close()


if __name__ == '__main__':
    main()
