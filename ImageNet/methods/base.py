import torch.nn as nn
from model import get_model, get_head

class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and head for training, evaluation function.
    """

    def __init__(self, cfg):
        super().__init__()
        self.model, self.out_size = get_model(cfg.arch,cfg)
        self.head = get_head(self.out_size, cfg)
        self.num_pairs = cfg.num_samples * (cfg.num_samples - 1) // 2
        self.add_head = cfg.add_head
        self.emb_size = cfg.emb
        self.cov_w = cfg.cov_w
        self.cov_stop = cfg.cov_stop
        self.rand_group = cfg.rand_group

    def forward(self, samples):
        raise NotImplementedError

    def step(self, progress):
        pass
    


