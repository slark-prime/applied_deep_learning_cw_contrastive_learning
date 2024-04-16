import random
import torch
import h5py


class H5ImageLoader():
    def __init__(self, img_file, batch_size, seg_file=None):
        self.img_h5 = h5py.File(img_file, 'r')
        self.dataset_list = list(self.img_h5.keys())
        if seg_file is not None:
            self.seg_h5 = h5py.File(seg_file, 'r')
            if set(self.dataset_list) > set(self.seg_h5.keys()):
                raise ("Images are not consistent with segmentation.")
        else:
            self.seg_h5 = None

        self.num_images = len(self.img_h5)
        self.batch_size = batch_size
        self.num_batches = int(self.num_images / self.batch_size)  # skip the remainders
        self.img_ids = [i for i in range(self.num_images)]
        self.image_size = self.img_h5[self.dataset_list[0]][()].shape

    def __iter__(self):
        self.batch_idx = 0
        random.shuffle(self.img_ids)
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        batch_img_ids = self.img_ids[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
        datasets = [self.dataset_list[idx] for idx in batch_img_ids]
        self.batch_idx += 1

        images = torch.stack(
            [torch.tensor(self.img_h5[ds][()]).permute(2, 0, 1).float() for ds in datasets])  # Convert and permute
        if self.seg_h5:
            labels = torch.stack([torch.tensor(self.seg_h5[ds][()]).long() for ds in datasets])
        else:
            labels = None

        return images, labels

    def __len__(self):
        return self.num_batches