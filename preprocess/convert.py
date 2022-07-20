import argparse
from collections import defaultdict
import os
import os.path as osp
import pandas as pd
from shutil import copyfile
import string
import random


def random_str(n=12):
    return ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(n)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', '-f', type=str, required=True)
    parser.add_argument('--video_folder', '-v', type=str, required=True)
    parser.add_argument('--audio_folder', '-a', type=str, required=True)
    parser.add_argument('--target_folder', '-t', type=str, required=True)
    args = parser.parse_args()

    # make target_folder
    if osp.exists(args.target_folder):
        raise ValueError(f'{args.target_folder} exists, please try a different target folder')
    os.makedirs(args.target_folder)
    target_video_folder = f'{args.target_folder}/videos'
    os.makedirs(target_video_folder)
    target_audio_folder = f'{args.target_folder}/audios'
    os.makedirs(target_audio_folder)

    # convert
    df = pd.read_csv(args.anno_file)
    result_dfs = defaultdict(list)
    for row_idx, row in df.iterrows():
        video_id = row.audio.rsplit('_', 2)[0]
        uuid = random_str()
        copyfile(f'{args.video_folder}/{row.listener}.mp4', f'{target_video_folder}/{uuid}.listener.mp4')
        copyfile(f'{args.video_folder}/{row.speaker}.mp4', f'{target_video_folder}/{uuid}.speaker.mp4')
        copyfile(f'{args.audio_folder}/{row.audio}.wav', f'{target_audio_folder}/{uuid}.wav')
        result_dfs[row.data_split].append({
            'video_id': video_id,
            'uuid': uuid,
            'speaker_id': row.speaker_id,
            'listener_id': row.listener_id,
        })
    
    for data_split, result_df in result_dfs.items():
        result_df = pd.DataFrame(result_df)
        result_df.to_csv(f'{args.target_folder}/{data_split}.csv', index=False)