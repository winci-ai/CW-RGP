# An Investigation into Whitening Loss for Self-supervised Learning

This is a PyTorch implementation of the paper.


## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install wandb for Logging ([wandb.ai](https://wandb.ai/)) 

## Experiments
The code includes experiments in section 4.1. 

#### Evaluation for Classification
The datasets include CIFAR-10, CIFAR-100, STL-10 and Tiny ImageNet, 
and the setup is strictly following [W-MSE paper](https://arxiv.org/abs/2007.06346).
The results of classification can be reproduced by the following commands:

```
#### CW-RGP 4
python train.py --dataset cifar10 --epochs 1000 --lr 3e-3 --num_samples 4 --bs 256 --w_size 64 --group 4
python train.py --dataset cifar100 --epochs 1000 --lr 3e-3 --num_samples 4 --bs 256 --w_size 64 --group 4
python train.py --dataset stl10 --epochs 2000 --lr 2e-3 --num_samples 4 --bs 256 --w_size 64 --group 4
python train.py --dataset tiny_in --epochs 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 1024 --w_size 128 --group 2 --head_size 2048


#### CW-RGP 2
python train.py --dataset cifar10 --epochs 1000 --lr 3e-3 --bs 256 --w_size 64 --group 4 --w_iter 4
python train.py --dataset cifar100 --epochs 1000 --lr 3e-3 --w_size 64 --group 4
python train.py --dataset stl10 --epochs 2000 --lr 2e-3 --w_size 64 --group 4
python train.py --dataset tiny_in --epochs 1000 --lr 2e-3 --emb 1024 --w_size 128 --group 2 --head_size 2048
```


Use `--no_norm` to disable normalization (for Euclidean distance).

#### Transfer to downstream tasks




<!--## Citation
```
@article{ermolov2020whitening,
  title={Whitening for Self-Supervised Representation Learning}, 
  author={Aleksandr Ermolov and Aliaksandr Siarohin and Enver Sangineto and Nicu Sebe},
  journal={arXiv preprint arXiv:2007.06346},
  year={2020}
}
```>
