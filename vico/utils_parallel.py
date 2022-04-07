from torch.optim import lr_scheduler
import os
import yaml

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory, exist_ok=True)
    return checkpoint_directory


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def write_log(log, output_directory):
    with open(os.path.join(output_directory, 'log.txt'), 'a') as f:
        f.write(log+'\n')


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictAverageMeter:
    def __init__(self, *keys):
        self.meters = {key: AverageMeter() for key in keys}

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key]
        raise AttributeError("has no attribute '{}'".format(key))

    def __getattr__(self, key):
        if key in self.meters:
            return self.meters[key]
        raise AttributeError("has no attribute '{}'".format(key))

    def update(self, val_dict):
        for key, values in val_dict.items():
            self.meters[key].update(values['val'], values.get('n', 1))

    def reset(self):
        for key in self.meters:
            self.meters[key].reset()
