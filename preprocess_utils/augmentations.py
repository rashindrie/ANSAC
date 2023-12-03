import numpy as np


def aug_rot(array, degrees, axes):
    """
    90 degree rotation.
    Args:
        array: batch in [b, x, y, c] format.
        degrees (int): rotation degree (0, 90, 180 or 270).
        axes: axes to apply the transformation.
    Returns: batch array.
    """

    if degrees == 0:
        pass
    elif degrees == 90:
        array = np.rot90(array, k=1, axes=axes)
    elif degrees == 180:
        array = np.rot90(array, k=2, axes=axes)
    elif degrees == 270:
        array = np.rot90(array, k=3, axes=axes)

    return array


def rot_flip_array(array, axes, rot_deg, flip):
    """
    Batch augmentation function supporting 90 degree rotations and flipping.
    Args:
        array: batch in [b, x, y, c] format.
        axes: axes to apply the transformation.
        rot_deg (int): rotation degree (0, 90, 180 or 270).
        flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
    Returns: batch array.
    """

    # Rot
    array = aug_rot(array, degrees=rot_deg, axes=axes)

    # Flip
    if flip == 'vertical':
        array = np.flip(array, axis=axes[0])
    elif flip == 'horizontal':
        array = np.flip(array, axis=axes[1])
    elif flip == 'both':
        array = np.flip(array, axis=axes[0])
        array = np.flip(array, axis=axes[1])
    elif flip == 'none':
        pass

    return array.copy()

