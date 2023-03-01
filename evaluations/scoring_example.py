# ranking-based scoring system based on an additive function
import numpy as np
import pandas as pd
import random
from string import ascii_uppercase
from collections import OrderedDict

# metric_name -> value for computation
larger_better = lambda e: -e
smaller_better = lambda e: e
zero_better = lambda e: abs(e)
metrics = OrderedDict([
    ('CPBD', larger_better),
    ('FID', smaller_better),
    ('SSIM', larger_better),
    ('PSNR', larger_better),
    ('CSIM', larger_better),
    ('ExpL1', smaller_better),
    ('PoseL1', smaller_better),
    ('AVOffset', zero_better),
    ('AVConf', larger_better),
    ('LipLMD', smaller_better),
    ('ExpFD', smaller_better),
    ('PoseFD', smaller_better),
])

# generate fake data
data = pd.DataFrame(
    [
        {'Name': name} | {k: random.random() if v != zero_better else random.random() - 0.5 for k, v in metrics.items()}
        for name in ascii_uppercase
    ]
)

# compute rank for each metric
data = data.assign(**{
    k + '_Rank': data[k].apply(v).rank().apply(int)
    for k, v in metrics.items()
})

# get sum of rank across all metrics
data = data.assign(Score=data[[k + '_Rank' for k in metrics.keys()]].sum(axis=1))

# remove columns with `_Rank`
data = data[[c for c in data.columns if '_Rank' not in c]]

# sort by rank and print
data = data.sort_values('Score')
print(data)
