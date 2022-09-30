#CW-RGP 4

#train 100 epochs with 4 GPUs
python ImageNet/train.py \
  --md \
  --imagenet_path [your imagenet-folder with train and val folders] \
  --cov_w 0.005 \ #optional, the weight of covariance loss
  --cov_stop 50 \ #optional, the epoch to cancel covariance loss


#train 200 epochs with 4 GPUs
python ImageNet/train.py \
  --epochs 200 \
  --md \
  --imagenet_path [your imagenet-folder with train and val folders] \
  --cov_w 0.01 \ #optional, the weight of covariance loss
  --cov_stop 100 \ #optional, the epoch to cancel covariance loss

#lincls
python ImageNet/lincls.py \
  --md \
  --pretrained [path to pretrained checkpoint] \
  --lars \
  --imagenet_path [your imagenet-folder with train and val folders] \