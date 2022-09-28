import torch.nn as nn
import torch.nn.functional as F
from methods.base import BaseMethod
from methods.losses import norm_mse_loss, cosine_loss


class SIMSIAM(BaseMethod):
    """ implements SimSiam https://arxiv.org/abs/2011.10566 """

    def __init__(self, cfg, loss='cos'):
        """ init additional target and predictor networks """
        super().__init__(cfg)
        self.head = nn.Sequential(
            nn.Linear(self.out_size, cfg.head_size, bias=False),
            nn.BatchNorm1d(cfg.head_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_size, cfg.head_size,bias=False),
            nn.BatchNorm1d(cfg.head_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_size, cfg.emb,bias=False),
            nn.BatchNorm1d(cfg.emb, affine=False),
        )
        self.pred = nn.Sequential(
            nn.Linear(cfg.emb, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.emb),
        )
        if loss == 'cos':
            self.loss_f = cosine_loss
        else:
            self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss

    def forward(self, samples):
        assert len(samples) == 2
        z = [self.head(self.model(x)) for x in samples]
        p = [self.pred(x) for x in z]

        z_stop = [x.detach() for x in z]

        loss = (self.loss_f(p[0], z_stop[1]) + self.loss_f(p[1], z_stop[0])) / 2
        return loss


