from openslide import OpenSlideError, open_slide
import openslide
import math
import PIL
from os.path import isfile, join, exists, dirname, isdir
from os import makedirs
import multiprocessing
from os import listdir

import numpy as np
import sklearn.cluster
import sys
from pathlib import Path

sys.path.append('../../')

from preprocess_utils.vectorize_wsi import SlideIterator

PIL.Image.MAX_IMAGE_PIXELS = 1000000000000

SCALE_FACTOR = 1
THUMBNAIL_FACTOR = 64
THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = "jpg"
DEST_TRAIN_EXT = "png"


def check_background(image, fn):
    pixels = image.getdata()  # get the pixels as a flattened sequence
    white_thresh = (240, 240, 240)
    nwhite = 0
    for pixel in pixels:
        if (pixel) >= white_thresh:
            nwhite += 1
    n = len(pixels)

    try:
        if (nwhite / float(n)) > 0.99:
            # image can not be used so return False since most of background is white
            return False
        if (nwhite / float(n)) < 0.2:
            # image can not be used so return False since most of background is black
            return False
    except:
        print(f"Error occurred for image: {fn} with {n} total  pixels and {nwhite} white pixels")
        return False
    return True


def open_wsi_slide(filepath):
    """
  Open a whole-slide image (*.svs, etc).

  Args:
    filepath: Path of the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
    try:
        slide = open_slide(filepath)
    except OpenSlideError:
        print("Error 1")
        slide = None
    except FileNotFoundError:
        print("Error 2")
        slide = None
    return slide


def slide_to_scaled_pil_image(filepath):
    """
  Convert a WSI training slide to a scaled-down PIL image.

  Args:
    filepath: The Path of the file.

  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """
    print("Opening Slide %s" % filepath)

    slide = open_wsi_slide(filepath)
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    try:
        whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        whole_slide_image = whole_slide_image.convert("RGB")
        img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        return img, large_w, large_h, new_w, new_h
    except:
        print("ERROR reading slide %s" % filepath)
    return None


def show_slide(filepath):
    """
  Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.

  Args:
    filepath: Path of the file
  """
    pil_img = slide_to_scaled_pil_image(filepath)[0]
    pil_img.show()


def get_training_slides(file_path):
    """
      Obtain the total number of WSI training slide images.

      Returns:
        The total number of WSI training slide images.
      """
    return [f for f in listdir(file_path) if (isfile(join(file_path, f)) and f.endswith(".svs"))]


def slide_info(file_path, display_all_properties=False):
    """
      Display information (such as properties) about training images.

      Args:
        display_all_properties: If True, display all available slide properties.
    """
    wsi_images = get_training_slides(file_path)
    obj_pow_20_list = []
    obj_pow_40_list = []
    obj_pow_other_list = []

    for slide in wsi_images:
        slide_filepath = file_path + slide
        slide = open_slide(slide_filepath)
        print("Level count: %d" % slide.level_count)
        print("Level dimensions: " + str(slide.level_dimensions))
        print("Level downsamples: " + str(slide.level_downsamples))
        print("Dimensions: " + str(slide.dimensions))
        objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        print("Objective power: " + str(objective_power))
        if objective_power == 20:
            obj_pow_20_list.append(slide)
        elif objective_power == 40:
            obj_pow_40_list.append(slide)
        else:
            obj_pow_other_list.append(slide)
        print("Associated images:")
        for ai_key in slide.associated_images.keys():
            print("  " + str(ai_key) + ": " + str(slide.associated_images.get(ai_key)))
        print("Format: " + str(slide.detect_format(slide_filepath)))
        if display_all_properties:
            print("Properties:")
            for prop_key in slide.properties.keys():
                print("  Property: " + str(prop_key) + ", value: " + str(slide.properties.get(prop_key)))

    print("\n\nSlide Magnifications:")
    print("  20x Slides: " + str(obj_pow_20_list))
    print("  40x Slides: " + str(obj_pow_40_list))
    print("  ??x Slides: " + str(obj_pow_other_list) + "\n")


def do_save_thumbnail(pil_img, size, path, display_path=False):
    """
  Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.

  Args:
    pil_img: The PIL image to save as a thumbnail.
    size:  The maximum width or height of the thumbnail.
    path: The path to the thumbnail.
    display_path: If True, display thumbnail path in console.
  """
    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    if display_path:
        print("Saving thumbnail to: " + path)
    dir = dirname(path)
    if dir != '' and not exists(dir):
        makedirs(dir)
    img.save(path)


def get_fov(name, img, thresh=220):
    # based on https://stackoverflow.com/questions/24169908/partitioning-images-based-on-their-white-space
    img_h, img_w = img.size

    # get thumbnail
    img = img.resize(
        (int(img_h / THUMBNAIL_FACTOR) * THUMBNAIL_FACTOR, int(img_w / THUMBNAIL_FACTOR) * THUMBNAIL_FACTOR))
    img_h, img_w = img.size

    img16 = img.resize((int(img_h / THUMBNAIL_FACTOR), int(img_w / THUMBNAIL_FACTOR)))

    dat = np.array(img16.convert(mode='L'))
    # img = np.array(img)

    h, w = dat.shape
    dat = dat.mean(axis=0)

    guesses = np.matrix(np.linspace(0, len(dat), 4)).T
    km = sklearn.cluster.KMeans(n_clusters=len(guesses), init=guesses, n_init=1)
    n_samples = np.matrix(np.nonzero(dat > thresh)).T

    whole_slide_image = img
    if (len(n_samples) > 0) and (len(n_samples) >= len(guesses)):
        km.fit(n_samples)
        c1, c2 = map(int, km.cluster_centers_[[1, 2]])
        img1 = img16.crop((0, 0, c1, h))
        img2 = img16.crop((c1, 0, c2, h))
        img3 = img16.crop((c2, 0, w, h))

        # save images:
        if check_background(img1, name):
            whole_slide_image = img.crop((0, 0, c1 * THUMBNAIL_FACTOR, h * THUMBNAIL_FACTOR))
            # whole_slide_image = PIL.Image.fromarray(img)
        elif check_background(img2, name):
            whole_slide_image = img.crop((c1 * THUMBNAIL_FACTOR, 0, c2 * THUMBNAIL_FACTOR, h * THUMBNAIL_FACTOR))
            # whole_slide_image = PIL.Image.fromarray(img)
        elif check_background(img3, name):
            whole_slide_image = img.crop((c2 * THUMBNAIL_FACTOR, 0, w * THUMBNAIL_FACTOR, h * THUMBNAIL_FACTOR))
            # whole_slide_image = PIL.Image.fromarray(img)
        else:
            print(f"No FOV found for {name} so using {h * THUMBNAIL_FACTOR}, {w * THUMBNAIL_FACTOR} at (0, 0)")
            whole_slide_image = img.crop((0, 0, c1 * THUMBNAIL_FACTOR, h * THUMBNAIL_FACTOR))

    return whole_slide_image


def training_slide_to_image(file_name, source_path, dest_path, dest_thumbnail, save_thumbnail=False):
    """
  Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.

  Args:
    slide_number: The slide number.
  """

    source_image_path = join(source_path, file_name)
    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(source_image_path)

    img_name = file_name.rpartition('.')[0].rpartition('/')[2] + '.' + DEST_TRAIN_EXT
    img_path = join(dest_path, img_name)

    # print("Saving image to: " + img_path)
    # img.save(img_path)

    fov_img = get_fov(file_name.rpartition('.')[0].rpartition('/')[2], img)
    # fov_img.save(img_path)
    # print(f'img: {img.size} fov:{fov_img.size}')

    del img

    thumbnail_path = join(dest_thumbnail, img_name)
    if save_thumbnail:
        do_save_thumbnail(fov_img, THUMBNAIL_SIZE, thumbnail_path)

    slide_directory = join(vector_path, img_name)
    Path(slide_directory).mkdir(parents=True, exist_ok=True)

    if not isfile(join(slide_directory, 'vectorised.txt')):
        print(f'Processing {slide_directory} with size {fov_img.size}')
        f = open(join(slide_directory, "vectorised.txt"), "w")
        f.close()

        # Read slide
        si = SlideIterator(
            image_path=None,
            threshold_mask=16,
            image=fov_img,
        )

        # Process it
        si.save_array(
            patch_size=256,
            stride=256,
            output_pattern=slide_directory + '/{item}',
            downsample=1
        )

    if isfile(join(slide_directory, 'patches.npy')):
        f = open(join(slide_directory, "vectorised.txt"), "w")
        f.write("Done")
        f.close()

    del fov_img


def training_slide_range_to_images(train_images, source_path, dest_path, dest_thumbnail):
    """
  Convert a range of WSI training slides to smaller images (in a format such as jpg or png).

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).

  """
    if not exists(dest_path):
        makedirs(dest_path)

    if not exists(dest_thumbnail):
        makedirs(dest_thumbnail)

    for slide in train_images:
        name = slide.rpartition('.')[0].rpartition('/')[2] + '.' + DEST_TRAIN_EXT
        if not isfile(join(dest_thumbnail, name)):
            temp_img = PIL.Image.new("RGB", (300, 300), (255, 255, 255))
            temp_img.save(join(dest_thumbnail, name))

            training_slide_to_image(slide, source_path, dest_path, dest_thumbnail, save_thumbnail=True)
    return len(train_images)


def singleprocess_training_slides_to_images(file_path):
    """
  Convert all WSI training slides to smaller images using a single process.
  """

    train_images = get_training_slides(file_path)
    training_slide_range_to_images(train_images)


def multiprocess_training_slides_to_images_extracted(source_path, dest_path, dest_thumbnail, converted_image_list=[]):
    """
  Convert all WSI training slides to smaller images using multiple processes (one process per core).
  Each process will process a range of slide numbers.
  """

    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    num_processes = 2
    pool = multiprocessing.Pool(num_processes)

    train_images = get_training_slides(source_path)

    converted_image_names = [i.rpartition('.')[0] for i in converted_image_list]
    train_image_names = [i.rpartition('.')[0] for i in train_images]

    final_list = []
    for img in train_image_names:
        if img not in converted_image_names:
            name = img + ".svs"
            final_list.append(name)

    train_images = final_list

    num_train_images = len(train_images)

    if num_train_images != 0:
        if num_processes > num_train_images:
            num_processes = num_train_images
        images_per_process = num_train_images / num_processes

        print("Number of processes: " + str(num_processes))
        print("Number of training images: " + str(num_train_images))

        # each task specifies a range of slides
        tasks = []
        for num_process in range(1, num_processes + 1):

            start_index = (num_process - 1) * images_per_process
            end_index = num_process * images_per_process

            start_index = int(start_index)
            end_index = int(end_index)

            tasks.append(train_images[start_index: end_index])

            if start_index == end_index:
                print("Task #" + str(num_process) + ": Process slide " + str(start_index))
            else:
                print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

        # start tasks
        results = []
        for t in tasks:
            results.append(
                pool.apply_async(training_slide_range_to_images, args=[t, source_path, dest_path, dest_thumbnail]))

        count = 0
        for result in results:
            count += result.get()
            if count == num_train_images:
                print("\nDone converting %d slides" % count)
            # else:
            # print("Done converting slides %d through %d" % (count, num_train_images))


original_slides = '/path/to/file'
converted_slides_path = '/path/to/file'
thumbnail_slides_path = '/path/to/file'
vector_path = '/path/to/file'

Path(vector_path).mkdir(parents=True, exist_ok=True)

# test on one slide training_slide_to_image('TCGA-A7-A26I-01B-06-BS6.E89976C5-6194-4D3A-82D3-1A8D1C3D88EF.svs',
# original_slides, converted_slides_path, thumbnail_slides_path, True)

multiprocess_training_slides_to_images_extracted(original_slides, converted_slides_path, thumbnail_slides_path)
