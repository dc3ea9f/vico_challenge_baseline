import argparse
import os
from scipy.io import loadmat
import glob
import os.path as osp
import numpy as np
import torch


def rearrange_mat(mat):
    coeff_3dmm = mat['coeff']
    _id = coeff_3dmm[:, :80]
    _exp = coeff_3dmm[:, 80:144]
    _tex = coeff_3dmm[:, 144:224]
    _angle = coeff_3dmm[:, 224:227]
    _gamma = coeff_3dmm[:, 227:254]
    _trans = coeff_3dmm[:, 254:257]
    _crop = mat['transform_params'][:, -3:]

    id_gamma_tex = np.concatenate((_id, _gamma, _tex), axis=1)
    angle_exp_trans = np.concatenate((_angle, _exp, _trans), axis=1)
    angle_exp_trans = np.concatenate((angle_exp_trans, _crop), axis=1)
    result = {
        'id.gamma.tex': id_gamma_tex,
        'angle.exp.trans': angle_exp_trans,
    }
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    mat_fns = glob.glob(f"{args.input_folder}/**/*.mat")
    os.makedirs(args.output_folder, exist_ok=True)
    for mat_fn in mat_fns:
        base_fn = osp.basename(mat_fn)[:-4]
        mat = loadmat(mat_fn)
        new_mat = rearrange_mat(mat)
        torch.save(new_mat, f'{args.output_folder}/{base_fn}.bin')
