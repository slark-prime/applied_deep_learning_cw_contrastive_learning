The `Finetune` directory contains Python scripts to fine-tune a ResNet model for image-related tasks:

- `Resnet.py`: Defines the ResNet model structure.
- `dataset.py`: Prepares the dataset, includes settings to adjust the training/validation split ratio.
- `finetune.py`: Contains the code for fine-tuning the pre-trained ResNet model. Modify the `ratio_train` variable inside to change the training proportion.
- `loader.py`: Loads the dataset from H5 files into PyTorch data loaders.
- `test.py`: Likely used for evaluating the fine-tuned model.

To run the fine-tuning process, navigate to the directory containing `finetune.py` in your command line and execute:

```shell
python finetune.py
```
-0.55 model:
-0.7 model:
-0.85 model: https://drive.google.com/file/d/1-NfKdLrcI_CIAOUoc1ozDn6A4ZRj6JT8/view?usp=sharing
