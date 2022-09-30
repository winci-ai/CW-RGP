# An Investigation into Whitening Loss for Self-supervised Learning

This is a PyTorch implementation of the paper.

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install wandb for Logging ([wandb.ai](https://wandb.ai/)) 

## Experiments
The code includes experiments in section 4.1. 

### Experimental Setup for Comparison of Baselines
The datasets include CIFAR-10, CIFAR-100, STL-10 and Tiny ImageNet, 
and the setup is strictly following [W-MSE paper](https://arxiv.org/abs/2007.06346).

the unsupervised pre-training scripts for small and medium datasets are shown in `scripts/base.sh`

The results are shown in the following table:

| Method   |CIFAR-10 | CIFAR-100 |STL-10 | Tiny-ImageNet |
| :--------:  |:-------------:| :--: | :--: | :--: |
|   | **top-1** &nbsp;&nbsp;&nbsp; **5-nn** |**top-1** &nbsp;&nbsp;&nbsp; **5-nn**  |**top-1** &nbsp;&nbsp;&nbsp; **5-nn** | **top-1** &nbsp;&nbsp;&nbsp; **5-nn** |
| CW-RGP 2|  91.92 &nbsp;&nbsp;&nbsp;   89.54 |  67.51 &nbsp;&nbsp;&nbsp;   57.35  |90.76 &nbsp;&nbsp;&nbsp;   87.34|  49.23 &nbsp;&nbsp;&nbsp;   34.04 |
| CW-RGP 4|  92.47 &nbsp;&nbsp;&nbsp; 90.74| 68.26 &nbsp;&nbsp;&nbsp;  58.67 |92.04 &nbsp;&nbsp;&nbsp; 88.95| 50.24 &nbsp;&nbsp;&nbsp;  35.99 |

### Experimental Setup for Large-Scale Classification

the unsupervised pre-training and linear classification scripts for ImageNet are shown in `scripts/ImageNet.sh`

#### Pre-trained Models
Our pre-trained ResNet-50 models:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">batch<br/>size</th>
<th valign="bottom">pre-train<br/>ckpt</th>
<th valign="bottom">linear cls.<br/>ckpt</th>
<th valign="center">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="center">100</td>
<td align="center">512</td>
<td align="center"><a href="https://drive.google.com/file/d/1p137aJGGtQIKc_UErx1F0IgUeEbhApS5/view?usp=sharing">train</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1xFsZjQZQ1SUPnhZ1MaZlODNjhqdNKW5h/view?usp=sharing">lincls</a></td>
<td align="center">69.7</td>
</tr>
<tr>
<td align="center">200</td>
<td align="center">512</td>
<td align="center"><a href="https://drive.google.com/file/d/1xMWmEW-AykQ5hdlfir0Tjjn8-UOOMHyx/view?usp=sharing">train</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1mqQS-YwbP7imf2LHRIp-wSmx-8AOjIAm/view?usp=sharing">lincls</a></td>
<td align="center">71.0</td>
</tr>
</tbody></table>

### Transferring to Object Detection
Same as [MoCo](https://github.com/facebookresearch/moco) for object detection transfer, please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).

| datasets |$AP_{50}$| $AP$ | $AP_{75}$ |ckpt|log|
| :----:  |:------:| :--: | :--: | :--: | :--: |
| VOC 07+12 detection  | $82.2_{±0.07}$|$57.2_{±0.10}$ | $63.8_{±0.11}$| [voc_ckpt](https://drive.google.com/file/d/1yUnBCCqcjBRhFJMi8R-cvnTIgqCUh7YB/view?usp=sharing)|[voc_log](https://drive.google.com/file/d/1tKUmBHUQiNZauiZ3Oe4-6YMsRG9iqILp/view?usp=sharing)|
| COCO detection| $60.5_{±0.28}$|$40.7_{±0.14}$ | $44.1_{±0.14}$|[coco_ckpt](https://drive.google.com/file/d/1ywNNEHGdX-ecztQV9nDFWN91Mu5cP1h6/view?usp=sharing) |[coco_log](https://drive.google.com/file/d/1ywNNEHGdX-ecztQV9nDFWN91Mu5cP1h6/view?usp=sharing)|
| COCO instance seg.| $57.3_{±0.16}$|$35.5_{±0.12}$ | $37.9_{±0.14}$|[coco_ckpt](https://drive.google.com/file/d/1ywNNEHGdX-ecztQV9nDFWN91Mu5cP1h6/view?usp=sharing) | [coco_log](https://drive.google.com/file/d/1ywNNEHGdX-ecztQV9nDFWN91Mu5cP1h6/view?usp=sharing)|

