import torch
import numpy as np
from PIL import Image

import preprocess_utils.augmentations as augmentations


class WsiNpySequence(object):
    def __init__(self, wsi_pattern, batch_size, rot_deg=0, flip='none', patch_size=768):
        """
        This class is used to make predictions on vectorized WSIs.
        Args:
            wsi_pattern (str): path pattern pointing to location of vectorized WSI.
                For example: "/path/SL000XXX/{item}.npy".
            batch_size (int): batch size to process the patches.
            rot_deg (int): rotation degree (0, 90, 180 or 270).
            flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
        """

        # Params
        self.batch_size = batch_size
        self.wsi_pattern = wsi_pattern
        self.rot_flip_fn = None
        self.rot_deg = None
        self.flip = None
        self.patch_size = patch_size

        # Read data
        self.image_tiles = np.load(wsi_pattern.format(item='patches'))
        self.xs = np.load(wsi_pattern.format(item='x_idx'))
        self.ys = np.load(wsi_pattern.format(item='y_idx'))
        self.image_shape = np.load(wsi_pattern.format(item='im_shape'))
        self.n_samples = self.image_tiles.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        # Set rot flip
        self.set_rot_flip(rot_deg, flip)

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def get_batch(self, idx):
        """
        Gets batches based on index. The last batch might have smaller length than batch size.
        Args:
            idx: index in batches..
        Returns: batch of image patches in [-1, +1] [b, x, y, ch] format.
        """

        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Build batch
        image_tiles = self.image_tiles[idxs, ...]

        # Format
        #         image_tiles = (image_tiles / 255.0 * 2) - 1

        return image_tiles

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        batch = self.rot_flip_fn(batch)
        batch = self.transform(batch)
        return batch

    def set_rot_flip(self, rot_deg, flip, axes=(1, 2)):
        """
        Sets the augmentation function applied to the entire batch.
        Args:
            rot_deg (int): rotation degree (0, 90, 180 or 270).
            flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
            axes (tuple): default=(1, 2), axis on which to apply the rot/flip (1,2) is for 3d feature maps
        """
        self.rot_deg = rot_deg
        self.flip = flip
        self.rot_flip_fn = lambda batch: augmentations.rot_flip_array(batch, axes=axes, rot_deg=rot_deg, flip=flip)

    def transform(self, batch):
        img_arr = np.zeros((batch.shape[0], self.patch_size, self.patch_size, 3))
        for idx in range(batch.shape[0]):
            im = batch[idx]
            im = Image.fromarray(im.astype('uint8'), 'RGB').resize((self.patch_size, self.patch_size))
            img_arr[idx] = np.array(im)

        out = normalize(img_arr)
        out = torch.from_numpy(out)

        return out


def normalize(input_rgb, size=768):
    VGG_MEAN = [103.939, 116.779, 123.68]  # bgr

    r, g, b = np.split(input_rgb, 3, 3)

    assert list(r.shape[1:]) == [size, size, 1]
    assert list(g.shape[1:]) == [size, size, 1]
    assert list(b.shape[1:]) == [size, size, 1]

    bgr = np.concatenate((
        b.astype(np.float32) - VGG_MEAN[0],
        g.astype(np.float32) - VGG_MEAN[1],
        r.astype(np.float32) - VGG_MEAN[2],
    ),
        axis=3
    )
    return bgr.transpose(0, 3, 1, 2)


def predict_generator(generator, feature_model, steps, device='cuda'):
    with torch.no_grad():
        for i in range(steps):
            q, _ = feature_model(generator[i].to(device))
            q = q.contiguous()
            q = q.view(q.shape[0], -1)
            q = q.detach().cpu().numpy()
            output = np.stack(q, axis=0)
            try:
                out = np.vstack((out, output))
            except:
                out = output

    return out
