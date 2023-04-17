import numpy as np
import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)


def jaccard_index(pred, target, threshold=0.5):
    pred_thresh = pred > threshold
    return torch.logical_and(pred_thresh, target).sum() / torch.logical_or(pred_thresh, target).sum()
