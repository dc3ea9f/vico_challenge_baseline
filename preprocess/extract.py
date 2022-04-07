import glob
import os
import os.path as osp
import argparse
from tqdm import tqdm
import librosa
import cv2
import torch
import numpy as np
from multiprocessing import Pool
import soundfile as sf
import torchaudio
from scipy.io import loadmat
torchaudio.set_audio_backend("sox_io")


def extract_audio_features(audio_fn, recons_folder, output_folder):
    video_id = osp.basename(audio_fn)[:-4]
    fps = 30

    audio, sr = sf.read(audio_fn)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    n_frames = len(loadmat(f'{recons_folder}/test/{video_id}.speaker.mat')['coeff'])
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0

    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]

        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)

        rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)

        curr_feat = np.concatenate((curr_mfccs, rms, zcr))
        curr_feats.append(curr_feat)
    curr_feats = np.stack(curr_feats, axis=0)
    with open(f'{output_folder}/{video_id}.npy', 'wb') as f:
        np.save(f, curr_feats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_folder', type=str, required=True)
    parser.add_argument('--input_recons_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    total_audio_fns = glob.glob(f'{args.input_audio_folder}/*.wav')
    
    for audio_fn in tqdm(total_audio_fns):
        extract_audio_features(audio_fn, args.input_recons_folder, args.output_folder)
