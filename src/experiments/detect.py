import typing as t

import numpy as np
from scipy.stats import binom

from src.utils.constants import *

DEFAULT_NFA_THRESHOLD = 10e-8


def detect_config(
    votes: np.ndarray,
    num_configs: int,
    start_position: t.Tuple[int, int] = (0,0),
    stop_position: t.Tuple[int, int] = (-1,-1),
    nfa_threshold: float = DEFAULT_NFA_THRESHOLD,
) -> int:
    """Detects the configuration of the demosaicing with respect to a NFA threshold.
    The votes can be eighter for the full config (algo, pattern), or on the algo only,
    or on the pattern only, or on the diagonal only.

    The detection is done only on a rectangular portion of the image.

    :param votes: The votes as integers
    :param num_configs: The number of possibility for each vote 
    :param start_position: The (x, y) coordinate of the upper-left corner of the region to analyse
    :param stop_position: The (x, y) coordinate of the lower-right corner of the region to analyse
    If (-1,-1), this will be extended to the lower-right corner of the image.
    :param nfa_threshold: The NFA to deceide whether or not the detection is relevant
    :returns: A tupple (best_config, minus_log_nfa) where:
        * `best_config` is the votes that got the majority on the analysed image
        * if the NFA of the detection is below `nfa_threshold`, `best_config` is -1.
        * minus_log_nfa = -log10(nfa)
    """
    # Parsing input
    height, width = votes.shape
    if stop_position == (-1, -1):
        stop_position = (height, width)
    assert stop_position[0] >=  start_position[0], 'Stop index < start index for axis 0'
    assert stop_position[1] >=  start_position[1], 'Stop index < start index for axis 1'

    # Extracting the portion on which to perform detection
    sub_votes = votes[
        start_position[0]:stop_position[0],
        start_position[1]:stop_position[1],
    ]
    num_votes = sub_votes.size

    # Getting the best config and its number of votes
    values, counts = np.unique(sub_votes, return_counts = True)
    best_index = np.argmax(counts)
    best_config = values[best_index]
    votes_for_best_config = counts[best_index]

    # Compute NFA
    nfa = num_configs * binom.sf(
        k=votes_for_best_config,
        n=num_votes,
        p = 1/num_configs,
    )

    # Output
    if nfa <= nfa_threshold:
        if nfa != 0:
            return (best_config, -np.log10(nfa))
        else:
            return (best_config, np.inf)
    else:
        return (-1, -np.log10(nfa))




    


