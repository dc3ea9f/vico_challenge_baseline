import sys
import glob
import numpy as np
import torch


def stas(x, start, end):
    cx = x[start:end]
    print(start, end, cx.max(), cx.min(), np.mean(cx))

if __name__ == '__main__':
    folder = sys.argv[1]
    mfcc_data = list(map(np.load, sorted(list(glob.glob(f"{folder}/audio_feats/*.npy")))))
    result = torch.from_numpy(np.concatenate(mfcc_data, axis=0))
    torch.save({'mean': torch.mean(result, dim=0), 'std': torch.std(result, dim=0)}, f'{folder}/audio_feats_mean_std.bin')

    dynam_data = list(map(lambda e: torch.load(e)['angle.exp.trans'], sorted(list(glob.glob(f"{folder}/video_feats/*.bin")))))
    fixed_data = list(map(lambda e: torch.load(e)['id.gamma.tex'], sorted(list(glob.glob(f"{folder}/video_feats/*.bin")))))

    dynam_data = torch.from_numpy(np.concatenate(dynam_data, axis=0))
    fixed_data = torch.from_numpy(np.concatenate(fixed_data, axis=0))

    result = {
        'angle.exp.trans': {'mean': torch.mean(dynam_data, dim=0), 'std': torch.std(dynam_data, dim=0)},
        'id.gamma.tex': {'mean': torch.mean(fixed_data, dim=0), 'std': torch.std(fixed_data, dim=0)},
    }

    torch.save(result, f'{folder}/video_feats_mean_std.bin')
