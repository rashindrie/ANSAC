"""
This module runs an encoder over a vectorized whole-slide image to obtain features from it (compress it).
"""

import os
import numpy as np
import matplotlib as mpl
from os.path import basename, join, exists, splitext


mpl.use('Agg')  # plot figures when no screen available

import preprocess_utils.augmentations as augmentations
from preprocess_utils.segment.seg_encode_utils import WsiNpySequence, predict_generator


def encode_wsi_npy_simple(encoder, wsi_pattern, batch_size, output_path=None, output_preview_pattern=None,
                          output_distance_map=False, save_features=False, device='cuda'):
    """
    Featurizes a vectorized whole-slide image using a pretrained encoder.
    Args:
        encoder: model transforming a patch to a vector code.
        wsi_pattern (str): path pattern pointing to vectorized WSI.
        batch_size (int): number of patches to encode simultaneously by the GPU.
        output_path (str): path pattern to output files.
            For example: /path/normal_001_features.npy'.
        output_preview_pattern (str or None): optional path pattern to preview files.
            For example: /path/normal_001_{f_min}_{f_max}_features.png'.
        output_distance_map (bool): True to write distance map useful to extract image crops.
        save_features (bool): True to write features 
    """

    # Read wsi
    wsi_sequence = WsiNpySequence(wsi_pattern=wsi_pattern, batch_size=batch_size)

    # Config
    xs = wsi_sequence.xs
    ys = wsi_sequence.ys
    image_shape = wsi_sequence.image_shape

    # Predict
    patch_features = predict_generator(generator=wsi_sequence, feature_model=encoder, steps=len(wsi_sequence), 
                                        device=device)
    features = np.ones((patch_features.shape[1], image_shape[1], image_shape[0])) * np.nan

    # Store each patch feature in the right spatial position
    for patch_feature, x, y in zip(patch_features, xs, ys):
        features[:, y, x] = patch_feature

    # Populate NaNs
    features[np.isnan(features)] = 0

    # Save to disk float16
    if save_features:
        np.save(output_path, features.astype('float16'))

#     # Plot
#     if output_preview_pattern:
#         plot_feature_map(np.copy(features), output_preview_pattern)

#     # Distance map
#     if output_distance_map:
#         try:
#             filename = splitext(basename(output_path))[0]
#             output_dm_path = join(dirname(output_path), filename + '_distance_map.npy')
#             distance_map = compute_single_distance_map(features.astype('float32'))
# #             np.save(output_dm_path, distance_map)
#             return features, distance_map
            
#         except Exception as e:
#             print('Failed to compute distance map for {f}. Exception: {e}.'.format(f=output_path, e=e), flush=True)
            
    return features


def encode_wsi_npy_advanced(encoder, code_size, wsi_pattern, batch_size, rot_deg, flip, output_pattern=None,
                            output_preview_pattern=None, output_distance_map=False, save_features=False, device='cuda', 
                            wsi_sequence_input=None):
    """
    Featurizes a vectorized whole-slide image using a pretrained encoder.
    Args:
        encoder: model transforming a patch to a vector code.
        wsi_pattern (str): path pattern pointing to vectorized WSI.
        batch_size (int): number of patches to encode simultaneously by the GPU.
        output_path (str): path pattern to output files.
            For example: /path/normal_001_features.npy'.
        rot_deg (int): rotation degree (0, 90, 180 or 270).
        flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
        output_preview_pattern (str or None): optional path pattern to preview files.
            For example: /path/normal_001_{f_min}_{f_max}_features.png'.
        output_distance_map (bool): True to write distance map useful to extract image crops.
        save_features (bool): True to write features 
    """

    output_npy_path = output_pattern.format(rot_deg=rot_deg, flip=flip)
    if output_preview_pattern:
        output_png_path = output_preview_pattern.format(rot_deg=rot_deg, flip=flip)
    else:
        output_png_path = None

    # Read wsi
    if wsi_sequence_input is None:
        wsi_sequence = WsiNpySequence(wsi_pattern=wsi_pattern, batch_size=batch_size)
    else:
        wsi_sequence = wsi_sequence_input

    # Config
    xs = wsi_sequence.xs
    ys = wsi_sequence.ys
    image_shape = wsi_sequence.image_shape

    # Prepare
    features = np.ones((code_size, image_shape[1], image_shape[0])) * np.nan
    idxs = np.arange(features.shape[1] * features.shape[2]).reshape((features.shape[1], features.shape[2]))

    # Augment
    idxs_rot = augmentations.rot_flip_array(idxs, axes=(0, 1), rot_deg=rot_deg, flip=flip)
    features = augmentations.rot_flip_array(features, axes=(1, 2), rot_deg=rot_deg, flip=flip)
    wsi_sequence.set_rot_flip(rot_deg, flip)

    # if isfile(output_npy_path):
    #     return features

    # # Save a dummy to path
    # if save_features:
    #     np.save(output_npy_path, [])

    # Predict
    patch_features = predict_generator(generator=wsi_sequence, feature_model=encoder, steps=len(wsi_sequence), 
                                        device=device)

    # Store each patch feature in the right position
    for patch_feature, x, y in zip(patch_features, xs, ys):
        idx = idxs[y, x]
        x_rot, y_rot = [ele for ele in zip(*np.where(idxs_rot == idx))][0]
        features[:, x_rot, y_rot] = patch_feature

    # Populate NaNs
    features[np.isnan(features)] = 0

    # Save to disk float16
    if save_features:
        np.save(output_npy_path, features.astype('float16'))

    # # Plot
    # if output_preview_pattern:
    #     plot_feature_map(np.copy(features), output_png_path)

    # # Distance map
    # if output_distance_map:
    #     try:
    #         filename = splitext(basename(output_npy_path))[0]
    #         output_dm_path = join(dirname(output_npy_path), filename + '_distance_map.npy')
    #         distance_map = compute_single_distance_map(features.astype('float32'))
    #         np.save(output_dm_path, distance_map)
    #         return features, distance_map
            
    #     except Exception as e:
    #         print('Failed to compute distance map for {f}. Exception: {e}.'.format(f=output_npy_path, e=e), flush=True)
            
    return features


def segment_augment_wsi_batch(wsi_pattern, encoder, output_dir, code_size, batch_size, aug_modes, save_features, overwrite=False):

    """
    Featurizes a vectorized whole-slide image given a set of augmentations (convenient wrapper
    for encode_wsi_npy_advanced() function).

    Args:
        wsi_pattern (str): path pattern pointing to location of vectorized WSI. For
        example: "/path/normal_060_{item}.npy".
        encoder: Keras model transforming a patch to a vector code.
        output_dir (str): output directory to store results.
        batch_size (int): batch size.
        aug_modes (list): list of pairs rotation-flipping values.
        overwrite (bool): True to overwrite existing files.
    """

    # Prepare paths
    if not exists(output_dir):
        os.makedirs(output_dir)

    filename = splitext(basename(wsi_pattern))[0]
    output_pattern = join(output_dir, filename.format(item='{rot_deg}_{flip}_features.npy'))
    output_preview_pattern = join(output_dir, filename.format(item='{rot_deg}_{flip}_features.png'))

    # Precheck
    process = False
    for flip, rot_deg in aug_modes:
        output_npy_path = output_pattern.format(rot_deg=rot_deg, flip=flip)
        if not exists(output_npy_path):
            process = True

    # Lock
    if process or overwrite:
        print('Featurizing image {image} ...'.format(image=wsi_pattern), flush=True)

        wsi_sequence = WsiNpySequence(wsi_pattern=wsi_pattern, batch_size=batch_size)

        # Iterate through augmentations
        for flip, rot_deg in aug_modes:
            # Encode
            try:
                print('Featurizing image {image} {rot_deg} {flip}...'.format(image=wsi_pattern, rot_deg=rot_deg, 
                    flip=flip), flush=True)
                encode_wsi_npy_advanced(
                    encoder=encoder,
                    code_size=code_size,
                    wsi_pattern=wsi_pattern,
                    wsi_sequence_input=wsi_sequence,
                    output_preview_pattern=output_preview_pattern,
                    output_pattern=output_pattern,
                    rot_deg=rot_deg,
                    flip=flip,
                    batch_size=batch_size,
                    save_features=save_features,
                    output_distance_map=True,
                )
            except Exception as e:
                print('Failed to encode {p} with rotation {r} and flip {f}. Exception: {e}'.format(p=output_pattern, r=rot_deg, f=flip, e=e), flush=True)

    else:
        print('Ignoring image {image} ...'.format(image=wsi_pattern), flush=True)
