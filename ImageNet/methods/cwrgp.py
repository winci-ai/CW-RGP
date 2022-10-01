import torch
import torch.nn.functional as F
from methods.whitening import Whitening2dZCA
from methods.base import BaseMethod
from methods.losses import norm_mse_loss, covariance_loss

class CW_RGP(BaseMethod):
    """ implements CW-RGP loss """
    """ Channel Whitening """

    def __init__(self, cfg):
        """ init whitening transform """
        super().__init__(cfg)
        self.whitening = Whitening2dZCA(track_running_stats=False,
                                        eps=cfg.w_eps,
                                        axis=cfg.axis,
                                        minus=cfg.minus,
                                        group=cfg.group)
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss
        self.w_iter = cfg.w_iter
        self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size
        self.axis = cfg.axis

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        if self.add_head:
            h = self.head(torch.cat(h))
        else:
            h = torch.cat(h)
        w_dim = h[0].size(-1)
        mse_loss = 0
        cov_loss = 0
        if self.cov_w != 0:
            for i in range(len(samples)):
                cov_loss += covariance_loss(h[i*bs:(i+1)*bs], self.axis)

        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(bs).view(-1, self.w_size)
            for idx in perm:
                for i in range(len(samples)):
                    if self.rand_group:
                        shuffle = torch.randperm(w_dim).tolist()
                    else: shuffle = None
                    z[idx + i * bs] = self.whitening(h[idx + i * bs], shuffle)
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    mse_loss += self.loss_f(x0, x1)
        mse_loss /= self.w_iter * self.num_pairs
        loss = mse_loss + self.cov_w * cov_loss
        return loss
    
    def step(self, epoch):
        if epoch >= self.cov_stop:
            self.cov_w = 0



