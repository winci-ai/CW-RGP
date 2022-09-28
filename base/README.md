# Self-Supervised Representation Learning

And 5 datasets:
- CIFAR-10 and CIFAR-100
- STL-10
- Tiny ImageNet
- ImageNet-100

Checkpoints are stored in `data` each 100 epochs during training.

The implementation is optimized for a single GPU, although multiple are also supported. It includes fast evaluation: we pre-compute embeddings for the entire dataset and then train a classifier on top. The evaluation of the ResNet-18 encoder takes about one minute.

## Installation

The implementation is based on PyTorch. Logging works on [wandb.ai](https://wandb.ai/). See `docker/Dockerfile`.

#### ImageNet-100
To get this dataset, take the original ImageNet and filter out [this subset of classes](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt). We do not use augmentations during testing, and loading big images with resizing on the fly is slow, so we can preprocess classifier train and test images. We recommend [mogrify](https://imagemagick.org/script/mogrify.php) for it. First, you need to resize to 256 (just like `torchvision.transforms.Resize(256)`) and then crop to 224 (like `torchvision.transforms.CenterCrop(224)`). Finally, put the original images to `train`, and resized to `clf` and `test`.

## Usage

Detailed settings are good by default, to see all options:
```
python -m train --help
python -m test --help
```

#### CW 4
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 512 --w_size 64 --group 4 --random "no_align"
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --num_samples 4 --bs 256 --emb 512 --w_size 64
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256
```

#### CW 2
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --w_size 128
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --w_size 256 --w_iter 4
```

#### Contrastive
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method contrastive
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --method contrastive
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method contrastive
```

#### BYOL
```
python -m train --dataset cifar10 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset cifar100 --epoch 1000 --lr 3e-3 --emb 64 --method byol
python -m train --dataset stl10 --epoch 2000 --lr 2e-3 --emb 128 --method byol
python -m train --dataset tiny_in --epoch 1000 --lr 2e-3 --emb 128 --method byol
```

#### ImageNet-100
```
python -m train --dataset imagenet --epoch 240 --lr 2e-3 --emb 128 --w_size 256 --crop_s0 0.08 --cj0 0.8 --cj1 0.8 --cj2 0.8 --cj3 0.2 --gs_p 0.2
python -m train --dataset imagenet --epoch 240 --lr 2e-3 --num_samples 4 --bs 256 --emb 128 --w_size 256 --crop_s0 0.08 --cj0 0.8 --cj1 0.8 --cj2 0.8 --cj3 0.2 --gs_p 0.2
```

Use `--no_norm` to disable normalization (for Euclidean distance).

## Citation
```
@article{ermolov2020whitening,
  title={Whitening for Self-Supervised Representation Learning}, 
  author={Aleksandr Ermolov and Aliaksandr Siarohin and Enver Sangineto and Nicu Sebe},
  journal={arXiv preprint arXiv:2007.06346},
  year={2020}
}
```
