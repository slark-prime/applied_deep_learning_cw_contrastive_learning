import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataset import prepare_dataset
from Resnet import resnet50x1
import torch.optim as optim
import torch.nn.functional as F
from loader import H5ImageLoader

DATA_PATH = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Ensure the directory for saving images exists
image_save_dir = './segmentation_results'
os.makedirs(image_save_dir, exist_ok=True)



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
    prepare_dataset(ratio_train=0.7)  # ratio_train can be chosen in [0, 0.85], as test set ratio is fixed at 10% #TODO: change training proportion

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
    num_epochs = 100
    criterion = DiceLoss()
    save_path = "/content/drive/MyDrive/ADL/results_pt"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## Data loader
    loader_train = H5ImageLoader(DATA_PATH + '/images_train.h5', minibatch_size, DATA_PATH + '/labels_train.h5')
    loader_val = H5ImageLoader(DATA_PATH + '/images_val.h5', 20, DATA_PATH + '/labels_val.h5')
    # loader_test = H5ImageLoader(DATA_PATH + '/images_test.h5', 20, DATA_PATH + '/labels_test.h5') #TODO: Debug

    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for param in model.parameters():
            param.requires_grad_(True)

        for batch_index, (images, masks) in enumerate(loader_train):
            images, masks = pre_process(images, masks)
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Add a channel dimension to masks if it's missing
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # Converts [N, H, W] to [N, 1, H, W]

            # print(f"Outputs shape: {outputs.shape}")
            # print(f"Masks shape (after unsqueeze if applied): {masks.shape}")

            # Ensure output dimensions match mask dimensions for height and width
            if outputs.shape[2:] != masks.shape[2:]:  # Ensure this references only spatial dimensions
                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            assert outputs.shape == masks.shape, f"Output shape {outputs.shape} doesn't match mask shape {masks.shape}"
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            # Print pixel values for debugging
            # if batch_index % 1000 == 0:  # Adjust the frequency of printing as needed
            #     print(f'Epoch {epoch+1}, Batch {batch_index}')
            #     print('Sample Image Pixel Values:', images[0, :, :5, :5].cpu().detach().numpy())
            #     print('Sample Mask Pixel Values:', masks[0, :, :5, :5].cpu().detach().numpy())
            #     print('Sample Output Pixel Values:', outputs[0, :, :5, :5].cpu().detach().sigmoid().numpy())


        avg_loss = total_loss / len(loader_train)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (images, masks) in enumerate(loader_val):
                images, masks = pre_process(images, masks)
                images, masks = images.to(device), masks.to(device)

                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)  # Converts [N, H, W] to [N, 1, H, W]

                outputs = model(images)
                if outputs.shape[2:] != masks.shape[2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

                assert outputs.shape == masks.shape, f"Output shape {outputs.shape} doesn't match mask shape {masks.shape}"
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Save images every 10 batches
                if batch_idx % 200 == 0:  # Adjust the frequency of saving images as needed
                    for i in range(min(images.size(0), 4)):  # Save up to 4 images per batch
                        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                        ax[0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
                        ax[0].set_title('Original Image')
                        ax[0].axis('off')

                        ax[1].imshow(masks[i].cpu().squeeze().numpy(), cmap='gray')
                        ax[1].set_title('True Mask')
                        ax[1].axis('off')

                        output = outputs[i].cpu().detach().squeeze().numpy()
                        ax[2].imshow(output, cmap='gray', vmin=0, vmax=1)  # Ensure correct visualization range
                        ax[2].set_title('Predicted Mask')
                        ax[2].axis('off')

                        plt.tight_layout()
                        filename = f'epoch_{epoch+1}_batch_{batch_idx}_image_{i}.png'
                        fig.savefig(os.path.join(image_save_dir, filename))
                        plt.close(fig)  # Close the figure to free memory

            avg_val_loss = val_loss / len(loader_val)
            print(f'Validation Loss: {avg_val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    print("Model saved.")


if __name__ == '__main__':
    main()