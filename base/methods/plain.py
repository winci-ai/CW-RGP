import torch
import torch.nn.functional as F
from methods.base import BaseMethod
from methods.losses import norm_mse_loss


class Plain(BaseMethod):
    """ implements MSE loss """

    def __init__(self, cfg):
        """ init whitening transform """
        super().__init__(cfg)
        self.whitening = torch.nn.Identity()
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss
        self.w_iter = cfg.w_iter
        self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        if self.add_head:
            h = self.head(torch.cat(h))
        else:
            h = torch.cat(h)
        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(bs).view(-1, self.w_size)
            for idx in perm:
                for i in range(len(samples)):
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs
        return loss


