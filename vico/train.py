from utils_parallel import prepare_sub_folder, write_log, get_config, DictAverageMeter, get_scheduler
from data import get_data_loader
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import os
import sys
import tensorboardX
import shutil
from box import Box
from logger import create_logger
import time
import datetime
import git
import torch.distributed as dist
import torch.optim as optim
from networks.listener_generator import ListenerGenerator
from networks.speaker_generator import SpeakerGenerator

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--config', type=str, default='configs/configs.yaml', help='Path to the config file.')
    parser.add_argument('--task', type=str, choices=['listener', 'speaker'], required=True)
    parser.add_argument('--time_size', type=int, required=True)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--anno_fn', type=str, default=None, help='specify anno file')
    parser.add_argument('--batch_size', type=int, default=None, help='specify batch size')
    parser.add_argument('--loss_weights', type=float, nargs=7, default=None, help='loss weights')
    parser.add_argument('--temporal_size', type=int, default=None, help='temporal size')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    return parser

def show_git_info():
    # get & show branch information and latest commit id
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch
    branch_name = branch.name
    latest_commit_sha = repo.head.object.hexsha
    print("Branch: {}".format(branch_name))
    print("Latest commit SHA: {}".format(latest_commit_sha))

if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()

    print("Use configs:", vars(opts))

    # Load experiment setting
    config = get_config(opts.config)
    config = Box(config)
    if opts.batch_size is not None:
        config.batch_size = opts.batch_size
    if opts.anno_fn is not None:
        config.anno_fn = opts.anno_fn
    if opts.loss_weights is not None:
        keys = sorted(list(config.loss_weights.keys()))
        new_dict = {k: opts.loss_weights[i] for i, k in enumerate(keys)}
        print('[WARNING] new loss weights:', new_dict)
        config.loss_weights = new_dict
    if opts.temporal_size is not None:
        config.temporal_size = opts.temporal_size
    if opts.max_epochs is not None:
        config.max_epochs = opts.max_epochs
    if opts.lr is not None:
        config.lr = opts.lr
    config.task = opts.task
    print('Using config:', config)

    # -- Init distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    opts.LOCAL_RANK = opts.local_rank
    torch.cuda.set_device(opts.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    config.LOCAL_RANK = opts.LOCAL_RANK
    torch.distributed.barrier()

    seed = opts.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    output_directory = os.path.join(opts.output_path)
    checkpoint_directory = prepare_sub_folder(output_directory)
    logger = create_logger(output_directory, dist.get_rank(), os.path.basename(output_directory.strip('/')))
    if dist.get_rank() == 0:
        shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    # -- Init data loader
    loader = get_data_loader(config, opts.task, opts.time_size)
    logger.info("Len loader: {}".format(len(loader)))

    # -- Init model
    if opts.task == 'listener':
        model = ListenerGenerator(config)
    else:
        model = SpeakerGenerator(config)
    if opts.resume is not None:
        print(f'resume model from {opts.resume}.')
        model.load_state_dict({k.replace('module.', '', 1): v for k, v in torch.load(opts.resume).items()})
    model = model.cuda()
    logger.info(str(model))

    # -- Init optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            betas=config.betas, weight_decay=config.weight_decay)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    lr_scheduler = get_scheduler(optimizer, config)

    meter = DictAverageMeter(*['iter_time', 'TOTAL_LOSS', 'loss_angle', 'loss_angle_spiky', 'loss_exp', 'loss_exp_spiky', 'loss_trans', 'loss_trans_spiky', 'loss_crop', 'loss_crop_spiky'])
    iterations = 0
    max_iter = config.max_epochs * len(loader)

    for epoch in range(config.max_epochs):
        meter.reset()
        lr_scheduler.step()
        loader.sampler.set_epoch(epoch)
        model.train()
        postfix = "%03d" % (epoch + 1)
        for id, data in enumerate(loader):
            start_iter = time.time()
            start_data = time.time()
            audio, driven_signal, init_signal, target_signal, lengths, _ = data

            audio = audio.cuda().float()
            driven_signal = driven_signal.cuda().float()
            init_signal = init_signal.cuda().float()
            target_signal = target_signal.cuda().float()
            lengths = lengths.cuda().long()

            elapse_data = time.time() - start_data

            # Main training code
            loss, loss_dict, pred_3dmm_dynam = model(
                audio,
                driven_signal,
                init_signal,
                lengths,
                target_signal,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapse_iter = time.time() - start_iter

            meter.update({
                'iter_time': {'val': elapse_iter},
                **loss_dict,
            })

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                etas = meter.iter_time.avg * (max_iter - iterations - 1)
                memory = torch.cuda.max_memory_allocated() / 1024. / 1024.
                loss_log = ''
                for loss_key, loss_item in loss_dict.items():
                    loss_log += f'{loss_key} {meter[loss_key].val:.4f}*{loss_item["weight"]} ({meter[loss_key].avg:.4f})\t'
                log = f'Train: [{iterations + 1}/{max_iter}]\teta {datetime.timedelta(seconds=int(etas))}\ttime {meter.iter_time.val:.2f} ({meter.iter_time.avg:.2f})\t'
                log += f'time_data {elapse_data:.2f}\t'
                log += loss_log
                log += f'mem {memory:.0f}MB'
                logger.info(log)
                write_log(log, output_directory)
            iterations += 1

        # Save network weights
        model_state_dict_name = os.path.join(checkpoint_directory, f'Epoch_{epoch + 1:03d}.bin')
        if dist.get_rank() == 0:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_state_dict_name)
        logger.info(f'save epoch {epoch + 1} model to {model_state_dict_name}')
