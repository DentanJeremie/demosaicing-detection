import typing as t

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.constants import *
from src.utils.datasets import no_noise_dataset
from src.utils.logs import logger
from src.utils.pathtools import project
from src.experiments.detect import detect_config
from src.experiments.forge import forge
from src.experiments.vote import (
    get_block_votes,
    get_block_votes_on_algo,
    get_block_votes_on_diag,
    get_block_votes_on_pattern,
)
NUM_NOFORGE = 1
NUM_ALGO = len(DEMOSAICING_ALGOS)
NUM_PATTERN = len(PATERNS)
NUM_DIAG = len(DIAGONALS)
NUM_CONFIG = NUM_NOFORGE + NUM_ALGO + NUM_PATTERN + NUM_DIAG
NUM_IMAGES = len(no_noise_dataset)
INDEX_TO_ALGO = {item:key for key, item in list(ALGO_TO_INDEX.items()) + [(None, -1)]}
INDEX_TO_PATTERN = {item:key for key, item in list(PATTERN_TO_INDEX.items()) + [(None, -1)]}

def do_evaluation_config_detection(
    nfa_threshold = None,
):
    """Does the evaluation of the configuration detection given a NFA threshold.

    :param nfa_threshold: The NFA threshold for the configuration detection.
    :returns: The list of false detections.
    """

    logger.info(f'Doing the evaluation of the configuration detection with NFA threshold = {nfa_threshold}')

    # Initializing the counts of true detections / false detections / no detections
    # true_detect = [array_for_no_jpeg, array_for_jpeg95, array_for_jpeg90]
    # array_for_jpeg95 is the count [true_detect_noforge, true_detect_bilinear, ..., true_detect_menon, true_detect_rggb, ..., true_detect_gbrg, true_detect_diag0, true_detect_diag1]
    # bar_bottom is just for stacking the bar charts true_detect, no_detect, and false_detect
    true_detect = [
        np.zeros(NUM_CONFIG)
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]
    false_detect = [
        np.zeros(NUM_CONFIG)
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]
    no_detect = [
        np.zeros(NUM_CONFIG)
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]
    bar_bottom = [
        np.zeros(NUM_CONFIG)
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]

    # Initializing the list of false detection configurations
    summary_false_detections = dict()

    # Filling the variables
    for image_index, original_image in enumerate(tqdm(no_noise_dataset)):
        for algo, pattern in [(None, None)] + ALGO_PATTERN_CONFIG:
            for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
                # Forging the image to impose the True configuration
                image = forge(
                    image=original_image,
                    demosaicing_algo=algo,
                    pattern=pattern,
                    jpeg_compression=jpeg_compression,
                )
                votes = get_block_votes(image)

                # Getting the votes on the algo, pattern, and diagonal
                votes_algo = get_block_votes_on_algo(votes)
                votes_pattern = get_block_votes_on_pattern(votes)
                votes_diag = get_block_votes_on_diag(votes)

                # Doing the detection
                detection_algo, _ = detect_config(votes_algo, num_configs=NUM_ALGO, nfa_threshold=nfa_threshold)
                detection_pattern, _ = detect_config(votes_pattern, num_configs=NUM_PATTERN, nfa_threshold=nfa_threshold)
                detection_diag, _ = detect_config(votes_diag, num_configs=NUM_DIAG, nfa_threshold=nfa_threshold)

                # Special case of (algo, pattern) == (None, None)
                if algo is None and pattern is None:
                    if detection_algo == -1 and detection_pattern == -1: # True detection
                        true_detect[jpeg_index][0] += 1/NUM_IMAGES
                    else:
                        false_detect[jpeg_index][0] += 1/NUM_IMAGES
                        summary_false_detections[f'image {image_index}: {algo}, {pattern}, {jpeg_compression}'] = (
                            f'nfa {nfa_threshold} image {image_index}: {algo}, {pattern}, {jpeg_compression} => {INDEX_TO_ALGO[detection_algo]}, {INDEX_TO_PATTERN[detection_pattern]}'
                        )
                    continue

                # Indexes of true algo, pattern, diagonal
                true_algo = ALGO_TO_INDEX[algo]
                true_pattern = PATTERN_TO_INDEX[pattern]
                true_diag = PATTERN_TO_DIAG_INDEX[pattern]

                # Filling algo's values
                if detection_algo == -1: # No detection
                    no_detect[jpeg_index][true_algo + NUM_NOFORGE] += 1/NUM_PATTERN/NUM_IMAGES
                elif detection_algo == true_algo: # True detection
                    true_detect[jpeg_index][true_algo + NUM_NOFORGE] += 1/NUM_PATTERN/NUM_IMAGES
                else: # False detection
                    false_detect[jpeg_index][true_algo + NUM_NOFORGE] += 1/NUM_PATTERN/NUM_IMAGES
                    summary_false_detections[f'image {image_index}: {algo}, {pattern}, {jpeg_compression}'] = (
                        f'nfa {nfa_threshold} image {image_index}: {algo}, {pattern}, {jpeg_compression} => {INDEX_TO_ALGO[detection_algo]}, {INDEX_TO_PATTERN[detection_pattern]}'
                    )

                # Filling pattern's values
                if detection_pattern == -1: # No detection
                    no_detect[jpeg_index][true_pattern + NUM_ALGO + NUM_NOFORGE] += 1/NUM_ALGO/NUM_IMAGES
                elif detection_pattern == true_pattern: # True detection
                    true_detect[jpeg_index][true_pattern + NUM_ALGO + NUM_NOFORGE] += 1/NUM_ALGO/NUM_IMAGES
                else: # False detection
                    false_detect[jpeg_index][true_pattern + NUM_ALGO + NUM_NOFORGE] += 1/NUM_ALGO/NUM_IMAGES
                    summary_false_detections[f'image {image_index}: {algo}, {pattern}, {jpeg_compression}'] = (
                        f'nfa {nfa_threshold} image {image_index}: {algo}, {pattern}, {jpeg_compression} => {INDEX_TO_ALGO[detection_algo]}, {INDEX_TO_PATTERN[detection_pattern]}'
                    )

                # Filling diag's values
                if detection_diag == -1: # No detection
                    no_detect[jpeg_index][true_diag + NUM_ALGO + NUM_PATTERN + NUM_NOFORGE] += 1/NUM_ALGO/2/NUM_IMAGES
                elif detection_diag == true_diag: # True detection
                    true_detect[jpeg_index][true_diag + NUM_ALGO + NUM_PATTERN + NUM_NOFORGE] += 1/NUM_ALGO/2/NUM_IMAGES
                else: # False detection
                    false_detect[jpeg_index][true_diag + NUM_ALGO + NUM_PATTERN + NUM_NOFORGE] += 1/NUM_ALGO/2/NUM_IMAGES
                    summary_false_detections[f'image {image_index}: {algo}, {pattern}, {jpeg_compression}'] = (
                        f'nfa {nfa_threshold} image {image_index}: {algo}, {pattern}, {jpeg_compression} => {INDEX_TO_ALGO[detection_algo]}, {INDEX_TO_PATTERN[detection_pattern]}'
                    )

    # Plotting
    GROUPS = ['Original'] + list(DEMOSAICING_ALGOS) + list(PATERNS) + list(DIAGONALS) 
    X_AXIS = np.arange(len(GROUPS))
    JPEG_BAR_OFFSET = [-0.20, 0.00, +0.20]

    # True detections
    for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
        if jpeg_index == 0:
            kwargs = dict(label='True detection')
        else:
            kwargs = dict()

        plt.bar(
            X_AXIS + JPEG_BAR_OFFSET[jpeg_index],
            true_detect[jpeg_index],
            width = 0.15,
            bottom=bar_bottom[jpeg_index],
            color='#2CD23E',
            **kwargs
        )
        bar_bottom[jpeg_index] += true_detect[jpeg_index]

    # No detection
    for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
        if jpeg_index == 0:
            kwargs = dict(label='No detection')
        else:
            kwargs = dict()

        plt.bar(
            X_AXIS + JPEG_BAR_OFFSET[jpeg_index],
            no_detect[jpeg_index],
            width = 0.15,
            bottom=bar_bottom[jpeg_index],
            color='#4149C3',
            **kwargs
        )
        bar_bottom[jpeg_index] += no_detect[jpeg_index]

    # False detection
    for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
        if jpeg_index == 0:
            kwargs = dict(label='False detection')
        else:
            kwargs = dict()

        plt.bar(
            X_AXIS + JPEG_BAR_OFFSET[jpeg_index],
            false_detect[jpeg_index],
            width = 0.15,
            bottom=bar_bottom[jpeg_index],
            color='#E53629',
            **kwargs
        )
        bar_bottom[jpeg_index] += false_detect[jpeg_index]

    # Saving figures
    plt.xticks(X_AXIS, [str(item)[:6] for item in GROUPS])
    plt.ylim(0, 1.35)
    plt.xlabel("Groups")
    plt.ylabel("Proportions of detections")
    plt.title(f"Detection with NFA threshold = {nfa_threshold}\n[Left, Middle, Right] = [None, JPEG95, JPEG90]")
    plt.legend()
    plt.savefig(project.output / f'config_detection_nfa_{nfa_threshold}.png')
    plt.close()

    return summary_false_detections


if __name__ == '__main__':
    sum_0 = do_evaluation_config_detection(1)
    sum_1 = do_evaluation_config_detection(1e-2)
    sum_2 = do_evaluation_config_detection(1e-8)

    full_sum = [
        item
        for summary in [sum_0, sum_1, sum_2]
        for _, item in summary.items()
    ]

    with (project.output / f'false_detection_summary.txt').open('w') as f:
        f.write('\n'.join(full_sum))