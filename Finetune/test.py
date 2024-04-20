import os
import torch
import torch.nn as nn
from dataset import prepare_dataset
from Resnet import resnet50x1
from loader import H5ImageLoader
import time
import torch.nn.functional as F
import numpy as np

DATA_PATH = './data'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, ps, ts):
        ps = torch.sigmoid(ps)  # Ensure predictions are in [0,1]
        ts = ts.float()  # Ensure targets are float for calculation

        if ps.shape != ts.shape:
            raise ValueError(f"Prediction shape {ps.shape} and target shape {ts.shape} do not match.")

        intersection = torch.sum(ts * ps, dim=(1, 2, 3))
        total = torch.sum(ts, dim=(1, 2, 3)) + torch.sum(ps, dim=(1, 2, 3))
        dice_score = (2. * intersection + self.eps) / (total + self.eps)

        return 1 - dice_score.mean()  # Return 1 - Dice to define loss


def pre_process(images, labels):
    # Convert images to PyTorch tensors and permute dimensions to [C, H, W]
    images = torch.stack([torch.tensor(img).permute(2, 0, 1).float() for img in images])

    # Ensure labels are also tensors; handle None and list of labels appropriately
    if labels is not None:
        labels = torch.stack([torch.tensor(lbl).long() for lbl in labels])
    else:
        labels = None  # Maintain None if no labels are present

    return images, labels

def main():
    # Prepare the dataset
    ratio_train = 0.7
    prepare_dataset(ratio_train)  # ratio_train can be chosen in [0, 0.85], as test set ratio is fixed at 10% #TODO: change training proportion

    model = resnet50x1()

    saved_model = "drive/MyDrive/best_model_0.7.pth"
    saved_model = torch.load(saved_model, map_location='cpu')

    ## Data loader
    loader_test = H5ImageLoader(DATA_PATH + '/images_test.h5', 20, DATA_PATH + '/labels_test.h5') 

    model.to(device)
    losses = []
    dsc_scores = []
    inf_times = []
    criterion = DiceLoss()
    im_size = (224,224)
    print("Start testing")
    for frames, masks in loader_test:
        frames, masks = pre_process(frames, masks)
        frames, masks = frames.to(device), masks.to(device)
        with torch.no_grad():
            start = time.time()
            output = model(frames)
            inf_times.append(time.time() - start)
            masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=output.shape[2:], mode='nearest').long()
            loss = criterion(output, masks_resized)
            losses.append(loss.item())
            dsc_scores.append(1 - loss.item())

    print(f"Average loss: {sum(losses) / len(losses)}")
    print(f"Average DSC score: {sum(dsc_scores) / len(dsc_scores)}")
    print(f"Average inference time: {sum(inf_times) / len(inf_times)}")

if __name__ == '__main__':
    main()