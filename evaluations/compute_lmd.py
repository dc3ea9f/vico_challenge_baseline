import pandas as pd
from glob import glob
import argparse
import cv2
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import face_alignment


def read_mp4(input_fn, to_rgb=False, to_gray=False, to_nchw=False):
    frames = []
    cap = cv2.VideoCapture(input_fn)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


if __name__ == '__main__':
    # predict landmark distance only for lips
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', type=str, required=True)
    parser.add_argument('--pd_video_folder', type=str, required=True)
    parser.add_argument('--anno_file', type=str, required=True)
    args = parser.parse_args()

    lmd_values = []
    lip_range = slice(48, 68)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    df = pd.read_csv(args.anno_file)
    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        gt_video_fn = f'{args.gt_video_folder}/{row.uuid}.speaker.mp4'
        pd_video_fn = f'{args.pd_video_folder}/{row.uuid}.speaker.mp4'

        assert osp.exists(gt_video_fn), f"'{gt_video_fn}' is not exist"
        assert osp.exists(pd_video_fn), f"'{pd_video_fn}' is not exist"

        gt_frames = read_mp4(gt_video_fn, True, False, False)
        pd_frames = read_mp4(pd_video_fn, True, False, False)
        for gt_frame, pd_frame in tqdm(zip(gt_frames, pd_frames), total=len(gt_frames)):
            gt_landmarks = fa.get_landmarks(gt_frame)
            pd_landmarks = fa.get_landmarks(pd_frame)
            
            if len(gt_landmarks) == 0:
                continue
            gt_landmarks, pd_landmarks = gt_landmarks[0][lip_range, :], pd_landmarks[0][lip_range, :]
            distances = np.abs(pd_landmarks - gt_landmarks)
            lmd_values.append(distances)
    print('lmd:', np.array(lmd_values).mean())
