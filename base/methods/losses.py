import torch.nn.functional as F
import torch


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


def cosine_loss(x0, x1):
    return -torch.nn.CosineSimilarity(dim=-1)(x0, x1).mean()


def covariance_loss(x: torch.tensor,axis):
    if axis == 0:
        x= x.T
    N, D = x.size()
    x = x - x.mean(dim=0)
    cov_x = (x.T @ x) / (N - 1)
    diag = torch.eye(D, device=x.device)
    cov_loss = cov_x[~diag.bool()].pow_(2).sum() / D
    return cov_loss

def contrastive_loss(x0, x1, tau, norm):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2
