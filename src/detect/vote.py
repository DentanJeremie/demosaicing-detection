import numpy as np
import skimage.measure

from src.utils.constants import *
from src.detect.forge import forge, get_forgery_configs

BLOCK_SHAPE = (2,2)


def get_block_votes(
    image: np.ndarray,
) -> np.ndarray:
    """Computes the vote of each 2x2 block. The vote is an int that represents
    the index of the configuration (algo, pattern) in `forge.get_forgery_configs()`
    that led to the minimal residual after the second democaising.

    :param image: A ndarray representing the input image, shape (height, width)
    :returns: An array of shape (height//2, width//2) whose values are ints representing 
    the vote of each 2x2 block.
    """
    # Iterating over the configurations algo, pattern
    stacked_residuals = []
    for algo, pattern in get_forgery_configs():
        forgery = forge(image, demosaicing_algo=algo, pattern=pattern, inplace=False)
        residual = np.abs(image - forgery)
        residual_one_channel = np.mean(residual, axis=2)
        residual_one_channel_by_block = skimage.measure.block_reduce(
            residual_one_channel,
            BLOCK_SHAPE,
            np.mean,
        )
        stacked_residuals.append(residual_one_channel_by_block)

    # Stacking and taking argmin of residual
    stacked_residuals = np.array(stacked_residuals)
    return np.argmin(stacked_residuals, axis=0)


def get_block_votes_on_algo(
    image: np.ndarray,
) -> np.ndarray:
    """Computes the vote on the algorithm of each 2x2 block. 
    The vote is an int that represents the index of the algo in `constants.DEMOSAICING_ALGOS`
    that led to the minimal residual after the second democaising.

    :param image: A ndarray representing the input image, shape (height, width)
    :returns: An array of shape (height//2, width//2) whose values are ints representing 
    the vote of each 2x2 block.
    """
    num_algos = len(DEMOSAICING_ALGOS)
    algo_to_index = {
        algo:index
        for index, algo in enumerate(DEMOSAICING_ALGOS)
    }

    # Iterating over the configs
    votes = get_block_votes(image)
    height, width = votes.shape
    votes_per_algo = np.zeros((height, width, num_algos))
    for index, (algo, _) in enumerate(get_forgery_configs()):
        votes_per_algo[algo_to_index[algo]] += (votes == index)

    return np.argmax(votes_per_algo, axis=2)


def get_block_votes_on_pattern(
    image: np.ndarray,
) -> np.ndarray:
    """Computes the vote on the pattern of each 2x2 block. 
    The vote is an int that represents the index of the algo in `constants.PATERNS`
    that led to the minimal residual after the second democaising.

    :param image: A ndarray representing the input image, shape (height, width)
    :returns: An array of shape (height//2, width//2) whose values are ints representing 
    the vote of each 2x2 block.
    """
    num_pattern = len(PATERNS)
    pattern_to_index = {
        pattern:index
        for index, pattern in enumerate(PATERNS)
    }

    # Iterating over the configs
    votes = get_block_votes(image)
    height, width = votes.shape
    votes_per_pattern = np.zeros((height, width, num_pattern))
    for index, (_, pattern) in enumerate(get_forgery_configs()):
        votes_per_pattern[pattern_to_index[pattern]] += (votes == index)

    return np.argmax(votes_per_pattern, axis=2)


def get_block_votes_on_diag(
    image: np.ndarray,
) -> np.ndarray:
    """Computes the vote on the diagonal of each 2x2 block. 
    The vote is an int that represents the index of the algo in `constants.DIAGONALS`
    that led to the minimal residual after the second democaising.

    :param image: A ndarray representing the input image, shape (height, width)
    :returns: An array of shape (height//2, width//2) whose values are ints representing 
    the vote of each 2x2 block.
    """
    num_diag = len(DIAGONALS)
    pattern_to_diag_index = {
        pattern:diag_index
        for diag_index, (_, pattern_list) in enumerate(DIAGONALS.items())
        for pattern in pattern_list
    }

    # Iterating over the configs
    votes = get_block_votes(image)
    height, width = votes.shape
    votes_per_diag = np.zeros((height, width, num_diag))
    for index, (_, pattern) in enumerate(get_forgery_configs()):
        votes_per_diag[pattern_to_diag_index[pattern]] += (votes == index)

    return np.argmax(votes_per_diag, axis=2)


if __name__ == '__main__':
    from src.utils.datasets import dataset
    original_image = dataset[0]
    forged_image = forge(
        original_image,
        demosaicing_algo='menon',
        pattern='BGGR'
    )
    jpeg_forged_image = forge(
        original_image,
        demosaicing_algo='menon',
        pattern='BGGR',
        prior_jpeg_compression=True,
        quality=0.3
    )
    votes_original = get_block_votes(original_image)
    votes_forged = get_block_votes(forged_image)
    votes_jpeg_forged = get_block_votes(jpeg_forged_image)
    print(np.unique(votes_original, return_counts=True))
    print(np.unique(votes_forged, return_counts=True))
    print(np.unique(votes_jpeg_forged, return_counts=True))