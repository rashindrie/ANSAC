import datetime
import math
import skimage.morphology as sk_morphology
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from os.path import join

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


class Time:
    """
  Class for displaying elapsed time.
  """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed


def np_info(np_arr, name=None, elapsed=None):
    """
  Display information (shape, type, max, min, etc) about a NumPy array.

  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
            name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def pil_to_np_rgb(pil_img):
    """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
    t = Time()
    rgb = np.asarray(pil_img)
    # np_info(rgb, "RGB", t.elapsed())
    return rgb


def mask_percent(np_img):
    """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        # print(
        #     "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        #         mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    # np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
    t = Time()
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    # np_info(result, "Filter Grays", t.elapsed())
    return result


def mask_rgb(rgb, mask):
    """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    # np_info(result, "Mask RGB", t.elapsed())
    return result


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
    t = Time()

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        # print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
        #     mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    # np_info(np_img, "Remove Small Objs", t.elapsed())
    return np_img


def apply_image_filters(original_tif):
    rgb = pil_to_np_rgb(original_tif)

    mask_not_green = filter_green_channel(rgb)
    # rgb_not_green = mask_rgb(rgb, mask_not_green)

    mask_not_gray = filter_grays(rgb, tolerance=20)
    # rgb_not_gray = mask_rgb(rgb, mask_not_gray)

    mask_gray_green_pens = mask_not_gray & mask_not_green
    # rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)

    mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, overmask_thresh=70, output_type="bool")
    rgb_remove_small = mask_rgb(rgb, mask_remove_small)

    # return Image.fromarray(rgb_remove_small)
    return rgb_remove_small


def get_patches_randomly(input_image, crop_size=224, final_size=224, patch_count=26, max_iterations=100,
                         foreground_threshold=20):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((final_size, final_size))

    h, w, c = input_image.shape
    count = 0
    patch_array = []

    rng = np.random.default_rng()
    random_numbers = rng.choice(h, size=max_iterations, replace=False)

    for x in random_numbers:
        if count == patch_count-1:
            break

        if (x + crop_size) < h and (x + crop_size) < w:
            patch = input_image[x:x + crop_size, x:x + crop_size, :]
            non_black_pixel = np.count_nonzero(patch)

            if ((non_black_pixel / patch.size) * 100) > foreground_threshold:
                count += 1

                if crop_size != final_size:
                    patch = resize(to_pil(patch))

                torch_patch = to_tensor(patch)

                patch_array.append(torch_patch)


    while len(patch_array) < patch_count-1:
        patch_array.append(torch.zeros((3, final_size, final_size)))

    resize_image = Image.fromarray(input_image).resize((final_size, final_size))
    patch_array.append(to_tensor(resize_image))

    stacked_patch_array = torch.stack([*patch_array], dim=0)

    return stacked_patch_array

def get_top_tiles_of_slide(patch_folder, final_size=224, patch_count=26, train=True, transform=None):
    to_tensor = transforms.ToTensor()

    patch_array = []

    patch_indices = np.arange(1, patch_count)

    if train:
        np.random.shuffle(patch_indices)

    for idx in patch_indices:
        patch_name = str(idx) + '.png'
        patch = Image.open(join(patch_folder, patch_name))

        if transform is not None:
            patch = transform(patch)

        patch_array.append(patch)

    full_image = Image.open(join(patch_folder, 'thumbnail.png'))

    if transform is not None:
        full_image = transform(full_image)
            
    patch_array.append(full_image)

    stacked_patch_array = torch.stack([*patch_array], dim=0)

    # print(stacked_patch_array.shape)

    return stacked_patch_array


def get_top_tiles_of_slide_2_magnifications(patch_folder, image_name='', final_size=224, patch_count=26, train=True, transform=None):
    # to_tensor = transforms.ToTensor()

    # patch_array = []

    # patch_indices = np.arange(1, int(patch_count/2))

    # if train:
    #     np.random.shuffle(patch_indices)

    # for idx in patch_indices:
    #     for j in [1,2]:
    #         patch_name = str(idx) + '_' + str(j) + '.png'
    #         patch = Image.open(join(patch_folder, patch_name))

    #         if transform is not None:
    #             patch = transform(patch)

    #         patch_array.append(patch)

    # stacked_patch_array = torch.stack([*patch_array], dim=0)

    # return stacked_patch_array

    ####################################################################

    # to_tensor = transforms.ToTensor()

    # patch_array = []

    # patch_indices = np.arange(1, int(patch_count/2))

    # if train:
    #     np.random.shuffle(patch_indices)

    # for idx in patch_indices:
    #     patch_name = str(idx) + '_2.png'
    #     patch = Image.open(join(patch_folder, patch_name))

    #     if transform is not None:
    #         patch = transform(patch)

    #     patch_array.append(patch)

    # for idx in patch_indices:
    #     patch_name = str(idx) + '_1.png'
    #     patch = Image.open(join(patch_folder, patch_name))

    #     if transform is not None:
    #         patch = transform(patch)

    #     patch_array.append(patch)

    # stacked_patch_array = torch.stack([*patch_array], dim=0)

    # return stacked_patch_array

    ####################################################################

    path_series_1 = '/data/gpfs/projects/punim1193/Cleopatra/data/series_1_patches'
    path_series_2 = '/data/gpfs/projects/punim1193/Cleopatra/data/series_2_patches'

    to_tensor = transforms.ToTensor()

    patch_array = []

    patch_indices = np.arange(1, int(patch_count/2))

    if train:
        np.random.shuffle(patch_indices)

    for idx in patch_indices:
        patch_name = join(image_name, str(idx) + '_2.png')
        patch = Image.open(join(path_series_2, patch_name))

        if transform is not None:
            patch = transform(patch)

        patch_array.append(patch)

    for idx in patch_indices:
        patch_name = join(image_name, str(idx) + '.png')
        patch = Image.open(join(path_series_1, patch_name))

        if transform is not None:
            patch = transform(patch)

        patch_array.append(patch)

    stacked_patch_array = torch.stack([*patch_array], dim=0)

    return stacked_patch_array


def mask_percent(np_img):
    """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def tissue_percent(np_img):
    """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
    return 100 - mask_percent(np_img)


def get_is_background(np_img, threshold):
    """
  Determine if np array is background or foreground

  Args:
    np_img: Image as a NumPy array.

  Returns:
    True if image is background and False if not
  """
    filtered_image = apply_image_filters(np_img)
    t_p = tissue_percent(filtered_image)

    if t_p > threshold:
        return False
    return True
    

def is_background_saltz(np_img):
    mean = np.mean(np_img[0,:,:]),np.mean(np_img[1,:,:]),np.mean(np_img[2,:,:])
    red, green, blue = np.std(np_img[0,:,:]),np.std(np_img[1,:,:]),np.std(np_img[2,:,:])
    
    sum_stds = (red + green + blue)/3
    
    if sum_stds < 18:
        return True
    return False
