# CW-RGP 4
# CIFAR-10
python base/train.py --dataset cifar10 --epochs 1000 --lr 3e-3 --num_samples 4 --bs 256 --w_size 64 --group 4

# CIFAR-100
python base/train.py --dataset cifar100 --epochs 1000 --lr 3e-3 --num_samples 4 --bs 256 --w_size 64 --group 4

# STL-10
python base/train.py --dataset stl10 --epochs 2000 --lr 2e-3 --num_samples 4 --bs 256 --w_size 64 --group 4

# Tiny-ImageNet
python base/train.py --dataset tiny_in --epochs 1000 --lr 2e-3 --num_samples 4 --bs 256 --emb 1024 --w_size 128 --group 2 --head_size 2048

# CW-RGP 2
# CIFAR-10
python base/train.py --dataset cifar10 --epochs 1000 --lr 3e-3 --bs 256 --w_size 64 --group 4 --w_iter 4

# CIFAR-100
python base/train.py --dataset cifar100 --epochs 1000 --lr 3e-3 --w_size 64 --group 4

# STL-10
python base/train.py --dataset stl10 --epochs 2000 --lr 2e-3 --w_size 64 --group 4

# Tiny-ImageNet
python base/train.py --dataset tiny_in --epochs 1000 --lr 2e-3 --emb 1024 --w_size 128 --group 2 --head_size 2048


#Use --no_norm to disable normalization (for Euclidean distance).