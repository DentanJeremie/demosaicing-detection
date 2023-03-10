import typing as t

import numpy as np
from scipy.stats import binom

from src.utils.constants import *
from src.utils.logs import logger


def detect_config(
    votes: np.ndarray,
    num_configs: int,
    start_position: t.Tuple[int, int] = (0,0),
    stop_position: t.Tuple[int, int] = (-1,-1),
    nfa_threshold: float = None,
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
        * `minus_log_nfa` = -log10(nfa)
    """
    # Parsing input
    height, width = votes.shape
    if stop_position == (-1, -1):
        stop_position = (height, width)
    if nfa_threshold is None:
        nfa_threshold = np.inf
    assert num_configs > 1, 'Impossible detection with num_configs <= 1.'
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


def detect_forgery(
    votes: np.ndarray,
    num_configs: int,
    windows_size: int,
    nfa_threshold: float = None,
    verbose: bool = False,
) -> int:
    """Detects if there is a demosaicing forgery with respect to a NFA threshold.
    The votes can be eighter for the full config (algo, pattern), or on the algo only,
    or on the pattern only, or on the diagonal only.

    :param votes: The votes as integers
    :param num_configs: The number of possibility for each vote 
    :param nfa_threshold: The NFA to deceide whether or not the detection is relevant
    :returns: A tupple (is_forgery, minus_log_nfa) where:
        * `is_forgery` is a boolean indicating if there is a forgery
        * `minus_log_nfa` = -log10(nfa)
    """
    # Parsing input
    if nfa_threshold is None:
        nfa_threshold = np.inf

    # Getting the global detection
    height, width = votes.shape
    global_detection, _ = detect_config(votes, num_configs, nfa_threshold=None)

    # List of start and stop positions of the window
    windows_start_stop_positions = [
        (
            (idx_0*windows_size//2, idx_1*windows_size//2),
            (idx_0*windows_size//2 + windows_size, idx_1*windows_size//2 + windows_size),
        )
        for idx_0 in range(2*height//windows_size)
        for idx_1 in range(2*width//windows_size)
        if idx_0*windows_size//2 + windows_size < height
        and idx_1*windows_size//2 + windows_size < width
    ]
    num_windows = len(windows_start_stop_positions)

    # Checking all window
    best_window_log_nfa = -np.inf
    for (start_position, stop_position) in windows_start_stop_positions:
        local_detection, local_log_nfa = detect_config(
            votes=votes,
            start_position=start_position,
            stop_position=stop_position,
            num_configs=num_configs,
            nfa_threshold=None,
        )
        if local_detection != global_detection and local_log_nfa > best_window_log_nfa:
            best_window_log_nfa = local_log_nfa
            if verbose:
                logger.info(f'Start position of the new best windows: {start_position}')
                logger.info(f'Stop position of the new best windows: {stop_position}')

    # Result
    best_window_log_nfa = -np.log10(num_windows*(num_configs - 1)/num_configs) + best_window_log_nfa

    if best_window_log_nfa != 0 and 10**(-best_window_log_nfa) <= nfa_threshold:
        return True, best_window_log_nfa
    else:
        return False, best_window_log_nfa
