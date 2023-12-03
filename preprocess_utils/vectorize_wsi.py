"""
This module extracts all valid patches from a mr-image and builds a numpy array with them for later fast processing.
Vectorized images are a requirement for compression.
https://github.com/davidtellez/neural-image-compression/blob/master/source/nic/vectorize_wsi.py
"""

# import multiresolutionimageinterface as mri  # see https://github.com/computationalpathologygroup/ASAP
import sys
import numpy as np
from PIL import Image
from matplotlib.image import imsave
from skimage.transform import downscale_local_mean
from os.path import dirname, join

sys.path.append('../')
sys.path.append('../../')

from preprocess_utils.scripts.utils import filters

Image.MAX_IMAGE_PIXELS = 10000000000000

class SlideIterator(object):
    def __init__(self, image_path, threshold_mask, image=None, im_resize=(20480, 20480)):
        self.image_path = image_path
        self.threshold_mask = threshold_mask
        self.image = image
        self.im_resize = im_resize
        self.load_data()
        
    def load_data(self):
        if self.image is None:
            self.image = Image.open(self.image_path)
        self.image = self.image.resize(self.im_resize)
        self.image = np.array(self.image)
        
        self.image_shape = self.image.shape
        
        self.mask = filters.apply_image_filters(self.image)
        
    def get_image_shape(self, stride):
        """
        Returns the image shape divided by the specified stride.
        Args:
            stride (int): pixels to ignore between patches.
        Returns: tuple or None.
        """

        if self.image_shape is not None:
            return (self.image_shape[0] // stride, self.image_shape[1] // stride)
        else:
            return None
        
    def iterate_patches(self, patch_size, stride, downsample=1):
        """
        Creates an iterator across valid patches in the mr-image (only non-empty mask patches qualify). It yields
        a tuple with:
            * Image patch: in [0, 255] uint8 [x, y, c] format.
            * Location of the patch in the image: index x divided by the stride.
            * Location of the patch in the image: index y divided by the stride.
        Args:
            patch_size (int): size of the extract patch.
            stride (int): pixels to ignore between patches.
            downsample (int): downsample patches and indexes (useful for half-level images).
        """
        
        # Iterate through all image patches
        self.feature_shape = self.get_image_shape(stride)
        
        y1 = 0
        y2 = self.image_shape[0]
        x1 = 0
        x2 = self.image_shape[1]
        
        for index_y in range(y1, y2, stride):
            for index_x in range(x1, x2, stride):
                # Avoid numerical issues by using the feature size
                if (index_x // stride >= self.feature_shape[0]) or (index_y // stride >= self.feature_shape[1]):
                    continue
                
                # Retrieve mask patch
                mask_tile = self.mask[index_y:index_y + stride, index_x:index_x + stride, :].astype('uint8')
                image_tile = self.image[index_y:index_y + stride, index_x:index_x + stride, :].astype('uint8')
                
                # Continue only if it is full of tissue.
                if not filters.get_is_background(mask_tile, self.threshold_mask):
                    # Downsample
                    if downsample != 1:
                        image_tile = downscale_local_mean(image_tile, (downsample, downsample, 1))
                        # image_tile = image_tile[::downsample, ::downsample, :]  # faster

                    # Yield
                    yield (image_tile, (index_x - x1) // stride, (index_y - y1) // stride)
                
    def save_array(self, patch_size, stride, output_pattern, downsample=1):
        """
        Iterates over valid patches and save them (and the indexes) as a uint8 numpy array to disk. This function
        writes the following files given an output pattern like '/path/normal_001_{item}.npy':
            * Patches: '/path/normal_001_patches.npy'
            * Indexes x: '/path/normal_001_x_idx.npy'
            * Indexes y: '/path/normal_001_y_idx.npy'
            * Image shape: '/path/normal_001_im_shape.npy'
            * Sanity check: '/path/normal_001_{item}.png'
        Args:
            patch_size (int): size of the extract patch.
            stride (int): pixels to ignore between patches.
            output_pattern (str): path to write output files.
            downsample (int): downsample patches and indexes (useful for half-level images).
        """

        # Paths
        safety_path = join(dirname(output_pattern), 'safety_image.png')
        
        # Iterate through patches
        image_tiles = []
        xs = []
        ys = []
        
        for image_tile, x, y in self.iterate_patches(patch_size, stride, downsample=downsample):
            image_tiles.append(image_tile)
            xs.append(x)
            ys.append(y)

#         # Concat
        image_tiles = np.stack(image_tiles, axis=0).astype('uint8')
        xs = np.array(xs)
        ys = np.array(ys)

        # Save image shape
        image_shape = self.get_image_shape(stride)

        # Store
        np.save(output_pattern.format(item='patches'), image_tiles)
        np.save(output_pattern.format(item='x_idx'), xs)
        np.save(output_pattern.format(item='y_idx'), ys)
        np.save(output_pattern.format(item='im_shape'), image_shape)

        # Safety check
        check_image = np.zeros(image_shape[::-1])
        for x, y in zip(xs, ys):
            check_image[y, x] = 1
        imsave(safety_path, check_image)


def vectorize_slide(image_path, threshold_mask, patch_size, stride, output_pattern, downsample=1, select_bounding_box=False):
    """
    Converts a whole-slide image into a numpy array with valid tissue patches for fast processing. It writes the
    following files for a given output pattern of '/path/normal_001_{item}.npy':
        * Patches: '/path/normal_001_patches.npy'
        * Indexes x: '/path/normal_001_x_idx.npy'
        * Indexes y: '/path/normal_001_y_idx.npy'
        * Image shape: '/path/normal_001_im_shape.npy'
        * Sanity check: '/path/normal_001_{item}.png'
    :param image_path: full path to whole-slide image file.
    :param output_pattern: full path to output files using the tag {item}. For example: '/path/normal_001_{item}.npy'.
    :param threshold_mask: ratio of tissue that must be present in the mask patch to qualify as valid.
    :param patch_size: size of the stored patches in pixels.
    :param stride: size of the stride used among patches in pixels (same as patch_size for no overlapping).
    :param downsample: integer indicating downsampling ratio for patches.
    :param select_bounding_box: True detect tissue within a bounding box (and discard the rest).
    :return: nothing.
    """
    # Read slide
    si = SlideIterator(
        image_path=image_path,
        threshold_mask=threshold_mask,
    )

    # Process it
    si.save_array(
        patch_size=patch_size,
        stride=stride,
        output_pattern=output_pattern,
        downsample=downsample
    )


