# Pretrain - PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

## Installation
```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Prepare Datasets (ImageNet)
ImageNet-1K data could be accessed with [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). The structure of Image Net folder should look like this:

```
root
├── data
│   ├── imagenet
│   │   ├── train
│   │   ├── val

```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./data/imagenet/train -dataset-name imageNet100 --log-every-n-steps 100 --epochs 100

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

