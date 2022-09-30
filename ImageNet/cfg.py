import argparse
from torchvision import models
from methods import METHOD_LIST
from functools import partial

DS_LIST = ["cifar10", "cifar100", "stl10", "tiny_in", "in100", "imagenet"]

def get_cfg():
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    """ generates configuration from user input in console """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument(
        "--bs", type=int, default=512, help="train bs",
    )
    parser.add_argument(
        "--bs_eval", type=int, default=1024, help="eval bs",
    )
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate for train', dest='lr')
    parser.add_argument('--lr_eval', '--learning-rate_eval', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate for lincls', dest='lr_eval')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('--wd_eval', '--weight-decay_eval', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay_eval')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
                    
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to checkpoint for evaluation(default: none)')

    parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed','--md', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
    parser.add_argument(
        "--method", type=str, choices=METHOD_LIST, default="cwrg", help="loss type",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CW-RGP",
        help="name of the run for wandb project",
    )
    parser.add_argument(
        "--cov_w", type=float, default=0, help="weight of covariance loss in CWRG"
    )
    parser.add_argument(
        "--cov_stop", type=int, default=50, help="the epoch to cancel the covariance loss in CWRG(set cov_w to 0)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="number of samples (d) generated from each image",
    )
    parser.add_argument(
        "--byol_tau", type=float, default=0.99, help="starting tau for byol loss"
    )
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

    parser.add_argument(
        "--w_eps", type=float, default=0, help="eps for stability for whitening"
    )
    parser.add_argument(
        "--head_layers", type=int, default=3, help="number of FC layers in head"
    )
    parser.add_argument(
        "--head_size", type=int, default=2048, help="size of FC layers in head"
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
        "--no_norm", dest="norm", action="store_false", help="don't normalize latents")
 
    parser.add_argument("--emb", type=int, default=1024, help="embedding size")

    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="imagenet")

    parser.add_argument(
        "--eval_head", action="store_true", help="eval head output instead of model",
    )
    parser.add_argument("--imagenet_path", type=str, default='data/ImageNet/')

    parser.add_argument("--axis", 
        type=int, 
        choices=[0,1],
        default=0, 
        help='0 for channel whitening, 1 for batch whitening')
    
    parser.add_argument(
        "--no_rg", dest="rand_group", action="store_false", help="don't use channel random group in whitening",
    )
    parser.add_argument("--minus", type=int, default=1, help='denominator n-x in calculating correlation matrix, default n-1')
    parser.add_argument("--group", type=int, default=2, help='number of whitening groups')



    return parser.parse_args()
