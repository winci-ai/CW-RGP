import torch
import torch.nn as nn
from methods.base import BaseMethod


class BARLOW_TWINS(BaseMethod):
    """ implements Barlow Twins https://arxiv.org/abs/2103.03230 """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bn = nn.BatchNorm1d(cfg.emb)
        self.lamb = cfg.barlow_lambda

    def forward(self, samples):
        assert len(samples) == 2

        z1 = self.head(self.model(samples[0]))
        z2 = self.head(self.model(samples[1]))

        c = self.bn(z1).T @ self.bn(z2)

        c.div_(len(z1[0]))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamb * off_diag
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

