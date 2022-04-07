from utils_parallel import prepare_sub_folder, write_log, get_config, DictAverageMeter, get_scheduler
from data_eval import get_data_loader
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
import os.path as osp
from scipy.io import savemat
import torchvision.transforms as transforms


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--config', type=str, default='configs/configs.yaml', help='Path to the config file.')
    parser.add_argument('--task', type=str, choices=['listener', 'speaker'], required=True)
    parser.add_argument('--anno_fn', type=str, default=None, help='specify anno file')
    parser.add_argument('--batch_size', type=int, default=None, help='specify batch size')
    parser.add_argument('--loss_weights', type=float, nargs=6, default=None, help='loss weights')
    parser.add_argument('--temporal_size', type=int, default=None, help='temporal size')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument('--SEED', type=int, default=42)
    return parser

def replace_coeffs(gt, new_value):
    result = {}
    result['id.gamma.tex'] = gt['id.gamma.tex']
    gt_angle_exp_trans = gt['angle.exp.trans'][:1, :]
    result['angle.exp.trans'] = np.concatenate((gt_angle_exp_trans, new_value), axis=0)
    return result

def convert_coeff_to_mat(coeff):
    _id, _gamma, _tex = np.split(coeff['id.gamma.tex'], (80, 27 + 80), axis=1) # (80, 27, 80)
    _angle, _exp, _trans, _crop = np.split(coeff['angle.exp.trans'], (3, 64 + 3, 3 + 64 + 3), axis=1) # (3, 64, 3, 3)
    coeffs = np.concatenate((_id, _exp, _tex, _angle, _gamma, _trans), axis=1)
    transform_params = np.concatenate((np.ones((_crop.shape[0], 2)) * 256., _crop), axis=1)
    result = {
        'coeff': coeffs,
        'transform_params': transform_params,
    }
    return result

def prepare_fake_mat(row, pred, *, task):
    if task == 'listener':
        gt_listener = torch.load(f'data/features/video_feats/{row.uuid}.listener.bin')
        fake_listener = replace_coeffs(gt_listener, pred)
        fake_listener_mat = convert_coeff_to_mat(fake_listener)
        return fake_listener_mat
    elif task == 'speaker':
        gt_speaker = torch.load(f'data/features/video_feats/{row.uuid}.speaker.bin')
        fake_speaker = replace_coeffs(gt_speaker, pred)
        fake_speaker_mat = convert_coeff_to_mat(fake_speaker)
        return fake_speaker_mat
    else:
        raise ValueError(f"got task='{task}'")

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
        config.loss_weights = new_dict
    if opts.temporal_size is not None:
        config.temporal_size = opts.temporal_size
    if opts.max_epochs is not None:
        config.max_epochs = opts.max_epochs
    config.task = opts.task
    print('Using config:', config)

    seed = opts.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    output_directory = os.path.join(opts.output_path)
    prepare_sub_folder(output_directory)
    logger = create_logger(output_directory, 0, os.path.basename(output_directory.strip('/')))
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    # -- Init data loader
    loader = get_data_loader(config, opts.task)
    logger.info("Len loader: {}".format(len(loader)))

    # -- Init model
    if opts.task == 'listener':
        model = ListenerGenerator(config)
    else:
        model = SpeakerGenerator(config)
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in torch.load(opts.resume).items()})

    model = model.cuda()
    logger.info(str(model))

    dynam_mean_std_info = torch.load('data/features/video_feats_mean_std.bin')['angle.exp.trans']
    dynam_mean, dynam_std = dynam_mean_std_info['mean'], dynam_mean_std_info['std']
    dynam_3dmm_transform = transforms.Lambda(lambda e: e * dynam_std + dynam_mean)

    mat_output_dir = osp.join(opts.output_path, 'recon_coeffs', 'test')
    os.makedirs(mat_output_dir, exist_ok=True)
    os.makedirs(osp.join(opts.output_path, 'vox_lmdb'), exist_ok=True)

    # test phase
    model.eval()
    test_in_id_count = 0
    with torch.no_grad():
        for id, data in enumerate(loader):
            start_iter = time.time()
            start_data = time.time()
            audio, driven_signal, init_signal, target_signal, lengths, rows = data

            audio = audio.cuda().float()
            driven_signal = driven_signal.cuda().float()
            init_signal = init_signal.cuda().float()
            target_signal = target_signal.cuda().float()
            lengths = lengths.cuda().long()

            elapse_data = time.time() - start_data

            # Main training code
            pred_3dmm_dynam = model(
                audio,
                driven_signal,
                init_signal,
                lengths,
            )
            target_signal = target_signal.cpu()
            
            for row, pred, target, length in zip(rows, pred_3dmm_dynam, target_signal, lengths):
                pred = pred[:length - 1].cpu()
                pred = dynam_3dmm_transform(pred)
                fake_mat = prepare_fake_mat(row, pred, task=opts.task)
                realname = row.uuid + '.' + opts.task
                savemat(osp.join(mat_output_dir, realname + '.mat'), fake_mat)
            logger.info(f"TestInId: write image\tepoch -1\tcount {test_in_id_count} / {len(loader)}")
            test_in_id_count += 1
