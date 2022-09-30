# An Investigation into Whitening Loss for Self-supervised Learning

This is a PyTorch implementation of the paper.


## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install wandb for Logging ([wandb.ai](https://wandb.ai/)) 

## Experiments
The code includes experiments in section 4.1. 

#### Experimental Setup for Comparison of Baselines
The datasets include CIFAR-10, CIFAR-100, STL-10 and Tiny ImageNet, 
and the setup is strictly following [W-MSE paper](https://arxiv.org/abs/2007.06346).

the unsupervised pre-training scripts for small and medium datasets are shown in `scripts/base.sh`

#### Experimental Setup for Large-Scale Classification

the unsupervised pre-training and linear classification scripts for ImageNet are shown in `scripts/ImageNet.sh`

| Method   |&nbsp;&nbsp;&nbsp;&nbsp; CIFAR-10 |  &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;CIFAR-100 |&nbsp;&nbsp;&nbsp;&nbsp; STL-10 |  &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;Tiny-ImageNet |
| :--------:  |:-------------:| :--: | :--: | :--: |
|   | top-1 &nbsp;&nbsp;&nbsp; 5-nn | &nbsp; &nbsp;top-1 &nbsp;&nbsp;&nbsp; 5-nn  |top-1 &nbsp;&nbsp;&nbsp; 5-nn | &nbsp; &nbsp;top-1 &nbsp;&nbsp;&nbsp; 5-nn  |
| **CW-RGP 2**|  **91.92** &nbsp;&nbsp;&nbsp;   **89.54** | &nbsp; &nbsp; **67.51** &nbsp;&nbsp;&nbsp;   **57.35**  |**90.76** &nbsp;&nbsp;&nbsp;   **87.34** | &nbsp; &nbsp; **49.23** &nbsp;&nbsp;&nbsp;   **34.04**  |
| **CW-RGP 4**|  **92.47** &nbsp;&nbsp;&nbsp; **90.74**| &nbsp; &nbsp; **68.26** &nbsp;&nbsp;&nbsp;  **58.67** |**92.04** &nbsp;&nbsp;&nbsp; **88.95**| &nbsp; &nbsp; **50.24** &nbsp;&nbsp;&nbsp;  **35.99** |

### Pre-trained Models
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
<td align="center"><a href="https://drive.google.com/file/d/1p137aJGGtQIKc_UErx1F0IgUeEbhApS5/view?usp=sharing">link</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1xFsZjQZQ1SUPnhZ1MaZlODNjhqdNKW5h/view?usp=sharing">link</a></td>
<td align="center">69.7</td>
</tr>
<tr>
<td align="center">200</td>
<td align="center">512</td>
<td align="center"><a href="https://drive.google.com/file/d/1xMWmEW-AykQ5hdlfir0Tjjn8-UOOMHyx/view?usp=sharing">link</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1mqQS-YwbP7imf2LHRIp-wSmx-8AOjIAm/view?usp=sharing">link</a></td>
<td align="center">71.0</td>
</tr>
</tbody></table>

### Transferring to Object Detection
Same as [MoCo](https://github.com/facebookresearch/moco) for object detection transfer, please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).


<!--## Citation
```
@article{ermolov2020whitening,
  title={Whitening for Self-Supervised Representation Learning}, 
  author={Aleksandr Ermolov and Aliaksandr Siarohin and Enver Sangineto and Nicu Sebe},
  journal={arXiv preprint arXiv:2007.06346},
  year={2020}
}
```>
