import io
import typing as t

import imageio
import numpy as np
from colour_demosaicing import (
    mosaicing_CFA_Bayer,
)

from src.utils.constants import *
from src.utils.logs import logger

COMPRESSION_FORMAT = 'jpeg'
COLOUR_RANGE = 255

def forge(
    image: np.ndarray,
    start_position: t.Tuple[int, int] = (0,0),
    stop_position: t.Tuple[int, int] = (-1,-1),
    demosaicing_algo: str = 'bilinear',
    pattern: str = 'RGGB',
    jpeg_compression: float = None,
) -> np.ndarray:
    """Forges an image by introducing democaising artefacts in a given zone of the image.
    More specifically, the function performs a mosaicing of the image and then a democaising
    in this zone, using a given pattern and demosaicing algorithm.

    The zone to forge is identified by two tupple of size 2: start_position and stop_position, and the forgery
    is performed in the rectangle defined by those tupples.

    :param image: The image to forge
    :param start_position: The first tupple to define the forgery zone
    :param stop_position: The second tupple to define the forgery zone. If (-1, -1), it will be modified to correspond
    to the bottom-right corner of the image, which is useful to forge the whole image.
    :param demosaicing_algo: A string identifying the demosaicing algo. Cf `forge.DEMOSAICING_ALGOS`
    If None, no demosaicing is done.
    :param pattern: A string identifying the pattern. Cf 'forge.PATTERNS`
    If None, no demosaicing is done.
    :param jpeg_compression: If None, no prior JPEG compression is applied after the forgery. 
    Else, a compression of quality `jpeg_compression` is applied.
    :returns: The forged image
    """
    logger.debug(f'Forging an image with algorithm {demosaicing_algo} and pattern {pattern}')

    # Parsing input
    height, width, channel = image.shape
    if stop_position == (-1, -1):
        stop_position = (height, width)

    assert channel == 3, 'Cannot forge an image that is not RGB-encoded.'
    assert stop_position[0] >=  start_position[0], 'Stop index < start index for axis 0'
    assert stop_position[1] >=  start_position[1], 'Stop index < start index for axis 1'
    assert demosaicing_algo in [None] + list(DEMOSAICING_ALGOS), 'Invalid demosaicing algo name'
    assert pattern in [None] + list(PATERNS), 'Invalid pattern name'

    # Selecting area to force
    area_to_forge = image[start_position[0]:stop_position[0],start_position[1]:stop_position[1],:]

    # Forging
    if demosaicing_algo is not None and pattern is not None:
        cfa = mosaicing_CFA_Bayer(area_to_forge, pattern=pattern)
        forged = DEMOSAICING_ALGOS[demosaicing_algo](cfa, pattern=pattern)
    else:
        forged = area_to_forge

    # Edditing the image
    result = image.copy()
    result[start_position[0]:stop_position[0],start_position[1]:stop_position[1],:] = forged

    # Clipping
    result = np.clip(result, a_min = 0.0, a_max = 1.0)

    # JPEG compression
    if jpeg_compression is not None:
        result = get_jpeg_compression(result, quality=jpeg_compression)

    return result


def get_jpeg_compression(
    image: np.ndarray,
    quality: float = 97,
) -> np.ndarray:
    """Performs a JPEG compression of the image."""
    logger.debug(f'JPEG compression with quality={quality}')
    # Compression
    compression_buffer = io.BytesIO()
    imageio.imwrite(
        compression_buffer,
        np.asarray(COLOUR_RANGE*image, np.uint8),
        format=COMPRESSION_FORMAT,
        quality = quality,
    )

    # Output
    return np.asarray(
        imageio.imread(compression_buffer, format=COMPRESSION_FORMAT),
        np.float32,
    ) / COLOUR_RANGE
