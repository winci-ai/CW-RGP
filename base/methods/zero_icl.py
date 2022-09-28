import torch
import torch.nn as nn
from methods.base import BaseMethod

class ZeroICL(BaseMethod):
    """ implements  """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, samples):
        assert len(samples) == 2

        z1 = self.head(self.model(samples[0]))
        z2 = self.head(self.model(samples[1]))

        loss = self.ins_loss(z1, z2)

        return loss

    def ins_loss(self, z1, z2):
        hidden = z1.size(1)
        whiten_net = WhitenTran()
        z1 = standardization(z1)
        z2 = standardization(z2)
        z1 = whiten_net.zca_forward(z1.transpose(0, 1))  # d * N
        z2 = whiten_net.zca_forward(z2.transpose(0, 1))
        c = torch.mm(z1.transpose(0, 1), z2)  # N * N

        c.div_(hidden)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        loss = on_diag
        return loss



def standardization(data, dim=-1):
    # N * d
    mu = torch.mean(data, dim=dim, keepdim=True)
    sigma = torch.std(data, dim=dim, keepdim=True)
    return (data - mu) / (sigma)


class WhitenTran(nn.Module):
    def __init__(self, eps=0.01, dim=256):
        super(WhitenTran, self).__init__()
        self.eps = eps
        self.dim = dim

    def pca_forward(self, x):
        """normalized tensor"""
        batch_size, feature_dim = x.size()
        f_cov = torch.mm(x.transpose(0, 1), x) / batch_size # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        f_cov_shrink = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrink.float()), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(feature_dim, feature_dim).detach()
        return torch.mm(inv_sqrt, x.transpose(0, 1)).transpose(0, 1)    # N * d

    def zca_forward(self, x):
        batch_size, feature_dim = x.size()
        f_cov = (torch.mm(x.transpose(0, 1), x) / (batch_size - 1)).float()  # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        # f_cov = torch.mm(x.transpose(0, 1), x).float()  # d * d
        U, S, V = torch.linalg.svd(0.9 * f_cov + 0.1 * eye)
        diag = torch.diag(1.0 / torch.sqrt(S+1e-5))
        rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach() # d * d
        return torch.mm(rotate_mtx, x.transpose(0, 1)).transpose(0, 1)  # N * d