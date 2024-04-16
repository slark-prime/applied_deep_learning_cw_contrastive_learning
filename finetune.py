import os

import torch
import torch.nn as nn
from dataset import prepare_dataset
from Resnet import resnet50x1
import torch.optim as optim

from loader import H5ImageLoader

DATA_PATH = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, ps, ts):
        # Convert predictions to probabilities and then to binary format
        ps = torch.sigmoid(ps)
        # ps = (ps > 0.5).float()

        # Ensure ts is a float tensor
        ts = ts.float()

        # Check and adjust dimensions if necessary
        if ps.shape != ts.shape:
            raise ValueError(f"Prediction shape {ps.shape} and target shape {ts.shape} do not match.")

        intersection = torch.sum(ts * ps, dim=(1, 2, 3))
        total = torch.sum(ts, dim=(1, 2, 3)) + torch.sum(ps, dim=(1, 2, 3))
        dice_score = (2 * intersection + self.eps) / (total + self.eps)

        return 1 - dice_score.mean()  # Average over the batch


def main():
    # Prepare the dataset
    prepare_dataset(ratio_train=0.7)  # ratio_train can be chosen in [0, 0.85], as test set ratio is fixed at 10%

    model = resnet50x1()
    model_dict = model.state_dict()

    sd = "/content/drive/MyDrive/ADL/resnet50-1x.pth"
    sd = torch.load(sd, map_location='cpu')
    filtered_pretrained_dict = {k: v for k, v in sd.items() if k in model_dict}
    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict)

    def initialize_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    model.apply(initialize_weights)

    ## Training parameters
    minibatch_size = 4
    learning_rate = 1e-4
    num_epochs = 2
    criterion = DiceLoss()
    save_path = "results_pt"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## Data loader
    loader_train = H5ImageLoader(DATA_PATH + '/images_train.h5', minibatch_size, DATA_PATH + '/labels_train.h5')
    loader_val = H5ImageLoader(DATA_PATH + '/images_val.h5', 20, DATA_PATH + '/labels_val.h5')
    # loader_test = H5ImageLoader(DATA_PATH + '/images_test.h5', 20, DATA_PATH + '/labels_test.h5')

    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in loader_train:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Add a channel dimension to masks if it's missing
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # Converts [N, H, W] to [N, 1, H, W]

            # print(f"Outputs shape: {outputs.shape}")
            # print(f"Masks shape (after unsqueeze if applied): {masks.shape}")

            # Ensure output dimensions match mask dimensions for height and width
            if outputs.shape[2:] != masks.shape[2:]:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear',
                                                          align_corners=False)

            assert outputs.shape == masks.shape, f"Output shape {outputs.shape} doesn't match mask shape {masks.shape}"
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader_train)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, masks in loader_val:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(loader_val)
            print(f'Validation Loss: {avg_val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    print("Model saved.")


if __name__ == '__main__':
    main()