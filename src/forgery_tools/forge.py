import io
import typing as t

import imageio
import numpy as np
from colour_demosaicing import (
    mosaicing_CFA_Bayer,
)

from src.utils.constants import *
from src.utils.datasets import no_noise_dataset
from src.utils.logs import logger

COMPRESSION_FORMAT = 'jpeg'
COLOUR_RANGE = 255
NUM_IMAGES = len(no_noise_dataset)

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


def generate_forged_image(jpeg_compression: float = None, verbose:bool = False) -> np.ndarray:
    """
    Generates an image presenting a forgery, i.e. an area in the image that is not demosaiced with the same
    algorithm as the rest of the image.
    * The image used is sample randomly in the `no_noise_dataset``
    * The global demosaicing algo and pattern of the image is sample randomly and enforced 
    with a first demosaicing execution
    * Then, within the forgery area (sampled randomly), another forgery execution is done 
    with another tuple (algo, pattern.
    * If asked, a jpeg compression is performed right before returning the forged image.

    :param jpeg_compression: If None, no compression is done.
    Else, `jpeg_compression` should be the quality of the desired compression.
    :returns: The forged image.
    """
    # Sampling the image
    image_index = np.random.randint(0, NUM_IMAGES)
    image = no_noise_dataset[image_index]
    height, width, _ = image.shape

    # Sampling demosaicing configurations
    config_index_0, config_index_1 = np.random.choice(range(len(ALGO_PATTERN_CONFIG)), 2, replace=False)
    (
        (global_algo, global_pattern),
        (forgery_algo, forgery_pattern)
    ) = ALGO_PATTERN_CONFIG[config_index_0], ALGO_PATTERN_CONFIG[config_index_1]

    # Sampling forgery zone
    start_position_0 = np.random.randint(0, height - FORGERY_WINDOWS_SIZE)
    start_position_1 = np.random.randint(0, width - FORGERY_WINDOWS_SIZE)
    start_position = (start_position_0, start_position_1)
    stop_position = (start_position_0 + FORGERY_WINDOWS_SIZE, start_position_1 + FORGERY_WINDOWS_SIZE)

    # Global config
    image = forge(
        image=image,
        demosaicing_algo=global_algo,
        pattern=global_pattern,
        jpeg_compression=None,
    )

    # Local forgery
    image = forge(
        image=image,
        start_position=start_position,
        stop_position=stop_position,
        demosaicing_algo=forgery_algo,
        pattern=forgery_pattern,
    )

    if verbose:
        logger.info(f'Creating a demosaicing forgery {global_algo, global_pattern} -> {forgery_algo, forgery_pattern}')
        logger.info(f'Forgery start and stop: {start_position} ->  {stop_position}')

    # JPEG compression
    if jpeg_compression is not None:
        image = get_jpeg_compression(image, quality=jpeg_compression)

    return image


def generate_unforged_image(jpeg_compression: float = None, verbose:bool = False) -> np.ndarray:
    """
    Generates an image without the same demosaicing on the whole image.
    If asked, a compression is done right before returning the result.

    :param jpeg_compression: If None, no compression is done.
    Else, `jpeg_compression` should be the quality of the desired compression.
    :returns: The forged image.
    """
    # Sampling the image
    image_index = np.random.randint(0, NUM_IMAGES)
    image = no_noise_dataset[image_index]
    height, width, _ = image.shape

    # Sampling demosaicing configurations
    config_index_0 = np.random.choice(range(len(ALGO_PATTERN_CONFIG)), replace=False)
    (global_algo, global_pattern) = ALGO_PATTERN_CONFIG[config_index_0]

    # Global config
    image = forge(
        image=image,
        demosaicing_algo=global_algo,
        pattern=global_pattern,
        jpeg_compression=None,
    )

    if verbose:
        logger.info(f'Creating a demosaicing-uniform image with config {global_algo, global_pattern}')

    # JPEG compression
    if jpeg_compression is not None:
        image = get_jpeg_compression(image, quality=jpeg_compression)

    return image