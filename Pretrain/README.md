# Pretrain - PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running SimCLR, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data <path to ImageNet dataset train folder> -dataset-name imageNet100 --log-every-n-steps 100 --epochs 100

```

If you want to run it on CPU (for debugging purposes) use the ```--disable-cuda``` option.

