import typing as t

import numpy as np
from numpy.random import RandomState

from src.utils.constants import *
from src.experiments.vote import get_block_votes_on_algo


def evaluate_algo_detection(
    images_and_labels: t.Iterable[t.Tuple[np.ndarray, int]]
) -> np.ndarray:
    """Evaluates the detection of the demosaicing algorithm.
    
    :param images_and_labels: An iterable whose values are tuples 
    (image, label) where `image` is the image as ndarray, and label is
    and int representing the """

    for image, label in images_and_labels:
        pass