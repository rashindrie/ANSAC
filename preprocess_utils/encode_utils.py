import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

import sys
sys.path.append('../')
import preprocess_utils.augmentations as augmentations


class WsiNpySequence(object):
    def __init__(self, wsi_pattern, batch_size, rot_deg=0, flip='none', attention=False):
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
        self.attention = attention

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
        if not self.attention:
            augmentation = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            out = torch.Tensor(batch.shape[0], 3, 224, 224)
            for idx in range(batch.shape[0]):
                im = batch[idx]
                im = Image.fromarray(im.astype('uint8'), 'RGB')
                t_img = augmentation(im)
                out[idx] = t_img
            return out

        else:
            augmentation = transforms.Compose([
                transforms.Resize((800, 800)),
            ])

            for idx in range(batch.shape[0]):
                im = batch[idx]
                im = Image.fromarray(im.astype('uint8'), 'RGB')
                im = augmentation(im)
                im = np.array(im)

                output = np.stack(im, axis=0)
                output = output[np.newaxis, ...]
                try:
                    out = np.vstack((out, output))
                except:
                    out = output

            return out


def predict_generator(generator, feature_model, steps, device='cuda'):
    with torch.no_grad():
        for i in range(steps):
            q = feature_model(generator[i].to(device))
            # q = nn.functional.normalize(q, dim=1)
            q = q.detach().cpu().numpy()
            output = np.stack(q, axis=0)
            try:
                out = np.vstack((out, output))
            except:
                out = output

    return out


def calc_score(batch, weight_stroma=1, weight_lym=1):
    for pb in batch[0]:
        (unique, counts) = np.unique(pb, return_counts=True)

        d = dict(zip(unique, counts))
        d.update(dict.fromkeys(set(range(6)).difference(d), 0))

        values = d.values()

        total = sum(values)

        background = d[0] / total
        tumor = d[1] / total
        stroma = d[2] / total
        lymphocyte = d[3] / total
        necrosis = d[4] / total
        other = d[5] / total

        sum_areas = (weight_stroma*stroma) + (weight_lym*lymphocyte) + 0*(tumor + background + necrosis + other)
        sum_areas = round(sum_areas, 4)
        try:
            out = np.vstack((out, sum_areas))
        except:
            out = sum_areas
    return out


def plot_feature_map(features, output_pattern):
    """
    Preview of the featurized WSI. Draws a grid where each small image is a feature map. Normalizes the set of feature
    maps using the 3rd and 97th percentiles of the entire feature volume. Includes these values in the filename.
    Args:
        features: numpy array with format [c, x, y].
        output_pattern (str): path pattern of the form '/path/tumor_001_90_none_{f_min:.3f}_{f_max:.3f}_features.png'
    """

    # Downsample to avoid memory error
    if features.shape[1] >= 800 or features.shape[2] >= 800:
        features = features[:, ::3, ::3]
    else:
        features = features[:, ::2, ::2]

    # Get range for normalization
    f_min = np.percentile(features[features != 0], 3)
    f_max = np.percentile(features[features != 0], 97)

    # Detect background (estimate)
    features[features == 0] = np.nan

    # Normalize and clip values
    features = (features - f_min) / (f_max - f_min + 1e-6)
    features = np.clip(features, 0, 1)

    # Add background
    features[features == np.nan] = 0.5

    # Make batch
    data = features[:, np.newaxis, :, :].transpose(0, 2, 3, 1)

    # Make grid
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.0)
    padding = ((0, 0), (5, 5), (5, 5)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.5)

    # Tile the individual thumbnails into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # Map the normalized data to colors RGBA
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1)
    image = cmap(norm(data[:, :, 0]))

    # Save the image
    plt.imsave(output_pattern, image)


def compute_single_distance_map(features):
    """
    Computes distance to tissue map. It is useful to detect where the tissue areas are located and take crops from them.
    :param features: featurized whole-slide image.
    :return: distance map array
    """

    # Binarize
    features = features.std(axis=0)
    features[features != 0] = 1

    # Distance transform
    distance_map = distance_transform_edt(features)
    distance_map = distance_map / np.max(distance_map)
    distance_map = np.square(distance_map)
    distance_map = distance_map / np.sum(distance_map)

    return distance_map

