# WSI-cycleGAN

Based on the Tensorflow implementation for learning an image-to-image translation **without** input-output pairs.
The method is proposed by [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) in 
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf). 

This code was forked from https://github.com/xhujoy/CycleGAN-tensorflow and modified to work optimally on Whole Slide images WSIs by [Brendon Lutnick](https://github.com/brendonlutnick)

You can download the pretrained model from [this url](https://1drv.ms/u/s!AroAdu0uts_gj5tA93GnwyfRpvBIDA)
and extract the rar file to `./checkpoint/`.


## Prerequisites
- tensorflow r1.1
- numpy 1.11.0
- scipy 0.17.0
- pillow 3.3.0

## Getting Started

### Train
- Download a dataset (e.g. zebra and horse images from ImageNet):
```bash
bash ./download_dataset.sh horse2zebra
```
- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra
```
- Use tensorboard to visualize the training details:
```bash
tensorboard --logdir=./logs
```

### Test
- Finally, test the model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=horse2zebra --phase=test --which_direction=AtoB
```

## Training and Test Details
To train a model,  
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ 
```
Models are saved to `./checkpoints/` (can be changed by passing `--checkpoint_dir=your_dir`).  

To test the model,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_dir=/path/to/data/ --phase=test --which_direction=AtoB/BtoA
```

## Datasets
This code takes 2 folders containing WSIs as an input. The code will pull random patches from the slides at runtime and attempt to translate bnetween the two sets. For example you could place H&E WSIs in folder A and PAS WSIs in folder B.

model.py __init__ contains 2 variables 
```bash
self.dataset_dirA
``` 
and 
```bash
self.dataset_dirB
```
which should be changed. 

The custom Tensorflow input pipeline used for this network can be downloaded [here](https://github.com/SarderLab/tf-WSI-dataset-utils) 


## Reference
- The torch implementation of CycleGAN, https://github.com/junyanz/CycleGAN
- The tensorflow implementation of pix2pix, https://github.com/yenchenlin/pix2pix-tensorflow
