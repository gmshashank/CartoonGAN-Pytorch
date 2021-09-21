# CartoonGAN-Pytorch
CartoonGAN: Generative Adversarial Networks for Photo Cartoonization

Pytorch testing code of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf) `[Chen et al., CVPR18]`. With the released pretrained [models](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm) by the authors, I made these simple scripts for a quick test.


## Getting started

- Linux
- NVIDIA GPU
- Pytorch 1.9

```
git clone https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch
cd CartoonGAN-Test-Pytorch-Torch
```

## Pytorch

The original pretrained models are Torch `nngraph` models, which cannot be loaded in Pytorch through `load_lua`. So I manually copy the weights (bias) layer by layer and convert them to `.pth` models. 

- Download the converted models:

```
sh pretrained_model/download_pth.sh
```

- For testing:

```
python test.py --input_dir YourImgDir --style Hosoda --gpu 0
```

## Note

- The training code should be similar to the popular GAN-based image-translation frameworks and thus is not included here.

## Acknowledgement

- Many thanks to the authors for this cool work.

    - Part of the codes are borrowed from [DCGAN](https://github.com/soumith/dcgan.torch), [TextureNet](https://github.com/DmitryUlyanov/texture_nets), [AdaIN](https://github.com/xunhuang1995/AdaIN-style), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch)

