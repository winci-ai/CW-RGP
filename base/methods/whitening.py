import abc
import torch
import torch.nn as nn


class Whitening2d(nn.Module):
    def __init__(self, momentum=0.01,
                 track_running_stats=True,
                 eps=0,
                 axis=0,
                 minus=1,
                 group=1,):
        super(Whitening2d, self).__init__()
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.axis = axis
        self.minus = minus
        self.group = group
        self.running_mean_registered = False
        self.running_variance_registered = False

    def forward(self, x, shuffle=None):
        # print(shuffle)
        assert self.axis in (0,1), "axis must be in (batch, channel) !"
        w_dim = x.size(-1)

        assert w_dim % self.group == 0, "The dim for whitening should be divisible by group !"

        m = x.mean(0 if self.axis == 1 else 1)
        m = m.view(1, -1) if self.axis == 1 else m.view(-1, 1)
        if self.track_running_stats:
            if not self.running_mean_registered:
                self.register_buffer(
                    "running_mean",
                    torch.zeros_like(m)
                )
                self.running_mean_registered = True
            if not self.training and self.axis == 1:  # for inference
                m = self.running_mean
        xn = x - m 

        sigma_dim = w_dim // self.group
        
        if self.axis == 1:
            eye = torch.eye(sigma_dim).type(xn.type()).reshape(1, sigma_dim, sigma_dim).repeat(self.group, 1, 1)  # [128, 128] / [64, 64]
        else:
            eye = torch.eye(x.size(0)).type(xn.type()).reshape(1, x.size(0), x.size(0)).repeat(self.group, 1, 1)

        if shuffle is None:
            shuffle = range(w_dim)

        if self.axis == 1:
            xn_g = xn[:, shuffle].reshape(-1, self.group, sigma_dim).permute(1, 0, 2)  # [4, 128, 16]
        else:
            xn_g = xn[:, shuffle].reshape(-1, self.group, sigma_dim).permute(1, 2, 0)  # [4, 64, 32]
        f_cov = torch.bmm(xn_g.permute(0, 2, 1), xn_g) / (xn_g.shape[1] - self.minus)  # [4, 16, 128] * [4, 128, 16] -> [4, 16, 16] / [4, 32, 64] * [4, 64, 32] -> [4, 32, 32]
        sigma = (1 - self.eps) * f_cov + self.eps * eye

        if self.track_running_stats:
            if not self.running_variance_registered:
                self.register_buffer(
                    "running_variance",
                    torch.eye(sigma_dim).reshape(1, sigma_dim, sigma_dim).repeat(self.group, 1, 1)
                )
                self.running_variance_registered = True
            if not self.training:  # for inference
                sigma = self.running_variance

        matrix = self.whiten_matrix(sigma, eye)  # [4, 16, 16] / [4, 32, 32]
        decorrelated = torch.bmm(xn_g, matrix)  # [4, 128, 16] * [4, 16, 16] -> [4, 128, 16] / [4, 64, 32] * [4, 32, 32] -> [4, 64, 32]

        shuffle_recover = [shuffle.index(i) for i in range(w_dim)]
        if self.axis == 1:
            decorrelated = decorrelated.permute(1, 0, 2).reshape(-1, w_dim)[:, shuffle_recover]
        else:
            decorrelated = decorrelated.permute(2, 0, 1).reshape(-1, w_dim)[:, shuffle_recover]

        if self.training and self.track_running_stats and self.axis == 0:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
                )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
                )

        return decorrelated

    @abc.abstractmethod
    def whiten_matrix(self, sigma, eye):
        pass

    def extra_repr(self):
        return "eps={}, momentum={}, axis={}, minus={}, group={}".format(
            self.eps, self.momentum, self.axis, self.minus, self.group)


class Whitening2dCholesky(Whitening2d):
    def whiten_matrix(self, sigma, eye):  # x [group, dim, dim]
        wm = torch.triangular_solve(
            eye, torch.cholesky(sigma), upper=False
        )[0]
        return wm.permute(0, 2, 1)


class Whitening2dZCA(Whitening2d):
    def whiten_matrix(self, sigma, eye):
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = torch.bmm(u, torch.diag_embed(scale))
        wm = torch.bmm(wm, u.permute(0, 2, 1))
        return wm


class Whitening2dPCA(Whitening2d):
    def whiten_matrix(self, sigma, eye):
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = torch.bmm(u, torch.diag_embed(scale))
        return wm





