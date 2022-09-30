from functools import partial
import argparse
from torchvision import models
import torch
import numpy as np
import random
from datasets import DS_LIST
from methods import METHOD_LIST


def get_cfg():
    """ generates configuration from user input in console """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--method", type=str, choices=METHOD_LIST, default="cwrg", help="loss type",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="ssl",
        help="name of the project for logging at https://wandb.ai",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CWRG",
        help="name of the run for wandb project",
    )
    parser.add_argument(
        "--byol_tau", type=float, default=0.99, help="starting tau for byol loss"
    )
    parser.add_argument('--barlow_lambda', default=0.008, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument("--vicreg_sim", default=25, type=float)
    parser.add_argument("--vicreg_var", default=25, type=float)
    parser.add_argument("--vicreg_cov", default=1.0, type=float)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="number of samples (d) generated from each image",
    )

    addf = partial(parser.add_argument, type=float)
    addf("--cj0", default=0.4, help="color jitter brightness")
    addf("--cj1", default=0.4, help="color jitter contrast")
    addf("--cj2", default=0.4, help="color jitter saturation")
    addf("--cj3", default=0.1, help="color jitter hue")
    addf("--cj_p", default=0.8, help="color jitter probability")
    addf("--gs_p", default=0.1, help="grayscale probability")
    addf("--crop_s0", default=0.2, help="crop size from")
    addf("--crop_s1", default=1.0, help="crop size to")
    addf("--crop_r0", default=0.75, help="crop ratio from")
    addf("--crop_r1", default=(4 / 3), help="crop ratio to")
    addf("--hf_p", default=0.5, help="horizontal flip probability")

    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=500,
        help="iterations of learning rate warmup",
    )
    parser.add_argument(
        "--no_add_bn", dest="add_bn", action="store_false", help="do not use BN in head"
    )
    parser.add_argument(
        "--no_head", dest="add_head", action="store_false", help="do not use head"
    )
    parser.add_argument("--knn", type=int, default=5, help="k in k-nn classifier")
    parser.add_argument("--fname", type=str, help="load model from file")
    parser.add_argument(
        "--lr_step",
        type=str,
        choices=["cos", "step", "none"],
        default="step",
        help="learning rate schedule type",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--eta_min", type=float, default=0, help="min learning rate (for --lr_step cos)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="weight decay (L2 penalty)"
    )
    parser.add_argument(
        "--optimizer", type=str, default='adam', help=""
    )
    parser.add_argument("--T0", type=int, help="period (for --lr_step cos)")
    parser.add_argument(
        "--Tmult", type=int, default=1, help="period factor (for --lr_step cos)"
    )
    parser.add_argument(
        "--w_eps", type=float, default=0, help="eps for stability for whitening"
    )
    parser.add_argument(
        "--head_layers", type=int, default=2, help="number of FC layers in head"
    )
    parser.add_argument(
        "--head_size", type=int, default=1024, help="size of FC layers in head"
    )

    parser.add_argument(
        "--w_size", type=int, default=128, help="size of sub-batch for W-MSE loss"
    )
    parser.add_argument(
        "--w_iter",
        type=int,
        default=1,
        help="iterations for whitening matrix estimation",
    )

    parser.add_argument(
        "--no_norm", dest="norm", action="store_false", help="don't normalize latents",
    )
    parser.add_argument(
        "--tau", type=float, default=0.5, help="contrastive loss temperature"
    )

    parser.add_argument("--epochs", type=int, default=200, help="total epoch number")
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
    
    
    parser.add_argument(
        "--eval_every_drop",
        type=int,
        default=5,
        help="how often to evaluate after learning rate drop",
    )
    parser.add_argument(
        "--eval_every", type=int, default=10, help="how often to evaluate"
    )
    parser.add_argument("--emb", type=int, default=512, help="embedding size")
    parser.add_argument(
        "--bs", type=int, default=512, help="number of original images in batch N",
    )
    parser.add_argument(
        "--drop",
        type=int,
        nargs="*",
        default=[50, 25],
        help="milestones for learning rate decay (0 = last epoch)",
    )
    parser.add_argument(
        "--drop_gamma",
        type=float,
        default=0.2,
        help="multiplicative factor of learning rate decay",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=[x for x in dir(models) if "resn" in x],
        default="resnet18",
        help="encoder architecture",
    )
    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="cifar10")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="dataset workers number",
    )
    parser.add_argument(
        "--clf",
        type=str,
        default="sgd",
        choices=["sgd", "knn", "lbfgs"],
        help="classifier for test.py",
    )
    parser.add_argument(
        "--eval_head", action="store_true", help="eval head output instead of model",
    )
    parser.add_argument("--imagenet_path", type=str, default="~/IN100/")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument("--axis", 
        type=int, 
        choices=[0,1],
        default=0, 
        help='0 for channel whitening, 1 for batch whitening')

    parser.add_argument("--minus", type=int, default=1, help='denominator n-x in calculating correlation matrix, default n-1')
    parser.add_argument("--group", type=int, default=1, help='number of whitening groups')
    parser.add_argument(
        "--no_rg", dest="rand_group", action="store_false", help="don't use channel random group in whitening",
    )
    parser.add_argument("--seed", type=int, default=1, help='rand seed')

    parser.add_argument("--gpu", type=str, default='0', help='which gpu to use')

    return parser.parse_args()

