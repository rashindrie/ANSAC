"""
This package contains utility functions to convert images between different
color spaces.
"""
# make functions available at the package level using these shadow imports
# since we mostly have one function per file
from .rgb_to_sda import rgb_to_sda
from .sda_to_rgb import sda_to_rgb

# list out things that are available for public use
__all__ = (

    # functions and classes of this package
    'sda_to_rgb',
    'rgb_to_sda',
)