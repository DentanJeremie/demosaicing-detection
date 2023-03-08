import numpy as np
import skimage.measure

from src.utils.constants import *
from src.experiments.forge import forge

BLOCK_SHAPE = (2,2)


def get_block_votes(
    image: np.ndarray,
) -> np.ndarray:
    """Computes the vote of each 2x2 block. The vote is an int that represents
    the index of the configuration (algo, pattern) in `constants.ALGO_PATTERN_CONFIG`
    that led to the minimal residual after the second democaising.

    :param image: A ndarray representing the input image, shape (height, width)
    :returns: An array of shape (height//2, width//2) whose values are ints representing 
    the vote of each 2x2 block.
    """
    # Iterating over the configurations algo, pattern
    stacked_residuals = []
    for algo, pattern in ALGO_PATTERN_CONFIG:
        forgery = forge(image, demosaicing_algo=algo, pattern=pattern)
        residual = np.abs(image - forgery)
        residual_one_channel = np.mean(residual, axis=2)
        residual_one_channel_by_block = skimage.measure.block_reduce(
            residual_one_channel,
            BLOCK_SHAPE,
            np.mean,
        )
        stacked_residuals.append(residual_one_channel_by_block)

    # Stacking and taking argmin of residual
    # We need random tie breaking for the argmax
    stacked_residuals = np.array(stacked_residuals)
    return np.argmin(
        1 - np.random.random(stacked_residuals.shape)
        * (stacked_residuals == np.min(stacked_residuals, axis=0)[np.newaxis, :, :]),
        axis=0
    )


def get_block_votes_on_algo(
    votes: np.ndarray,
) -> np.ndarray:
    """Computes the vote on the algorithm of each 2x2 block. 
    The vote is an int that represents the index of the algo in `constants.DEMOSAICING_ALGOS`
    that led to the minimal residual after the second democaising.

    :param votes: A ndarray representing the votes on theinput image, shape (height, width)
    :returns: An array of shape (height, width) whose values are ints representing 
    the vote of each 2x2 block.
    """
    num_algos = len(DEMOSAICING_ALGOS)

    # Iterating over the configs
    height, width = votes.shape
    votes_per_algo = np.zeros((num_algos, height, width))
    for index, (algo, _) in enumerate(ALGO_PATTERN_CONFIG):
        votes_per_algo[ALGO_TO_INDEX[algo]] += (votes == index)

    return np.argmax(votes_per_algo, axis=0)


def get_block_votes_on_pattern(
    votes: np.ndarray,
) -> np.ndarray:
    """Computes the vote on the pattern of each 2x2 block. 
    The vote is an int that represents the index of the algo in `constants.PATERNS`
    that led to the minimal residual after the second democaising.

    :param votes: A ndarray representing the votes on the input image, shape (height, width)
    :returns: An array of shape (height, width) whose values are ints representing 
    the vote of each 2x2 block.
    """
    num_pattern = len(PATERNS)

    # Iterating over the configs
    height, width = votes.shape
    votes_per_pattern = np.zeros((num_pattern, height, width))
    for index, (_, pattern) in enumerate(ALGO_PATTERN_CONFIG):
        votes_per_pattern[PATTERN_TO_INDEX[pattern]] += (votes == index)

    return np.argmax(votes_per_pattern, axis=0)


def get_block_votes_on_diag(
    votes: np.ndarray,
) -> np.ndarray:
    """Computes the vote on the diagonal of each 2x2 block. 
    The vote is an int that represents the index of the algo in `constants.DIAGONALS`
    that led to the minimal residual after the second democaising.

    :param image: A ndarray representing the votes on the input image, shape (height, width)
    :returns: An array of shape (height, width) whose values are ints representing 
    the vote of each 2x2 block.
    """
    num_diag = len(DIAGONALS)

    # Iterating over the configs
    height, width = votes.shape
    votes_per_diag = np.zeros((num_diag, height, width))
    for index, (_, pattern) in enumerate(ALGO_PATTERN_CONFIG):
        votes_per_diag[PATTERN_TO_DIAG_INDEX[pattern]] += (votes == index)

    return np.argmax(votes_per_diag, axis=0)


if __name__ == '__main__':
    from src.utils.datasets import no_noise_dataset
    original_image = no_noise_dataset[0]
    forged_image = forge(
        original_image,
        demosaicing_algo='menon',
        pattern='BGGR'
    )
    votes_original = get_block_votes(original_image)
    votes_forged = get_block_votes(forged_image)
    print(np.unique(votes_original, return_counts=True))
    print(np.unique(votes_forged, return_counts=True))
    print(np.unique(get_block_votes_on_algo(votes_original), return_counts=True))
    print(np.unique(get_block_votes_on_algo(votes_forged), return_counts=True))