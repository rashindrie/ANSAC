"""
This package contains utility functions that are widely used by functions in
all other sub-packages of histomicstk
"""

# make functions available at the package level using shadow imports
# since we mostly have one function per file
from .convert_image_to_matrix import convert_image_to_matrix
from .convert_matrix_to_image import convert_matrix_to_image
from .exclude_nonfinite import exclude_nonfinite

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'convert_matrix_to_image',
    'convert_image_to_matrix',
    'exclude_nonfinite'
)
