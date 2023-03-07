import typing as t

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.constants import *
from src.utils.datasets import no_noise_dataset
from src.experiments.detect import detect_config
from src.experiments.forge import forge
from src.experiments.vote import (
    get_block_votes,
    get_block_votes_on_algo,
    get_block_votes_on_diag,
    get_block_votes_on_pattern,
)
NUM_ALGO = len(DEMOSAICING_ALGOS)
NUM_PATTERN = len(PATERNS)
NUM_DIAG = len(DIAGONALS)
NUM_CONFIG = NUM_ALGO + NUM_PATTERN + NUM_DIAG

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

saving_done = False

# Filling the variables
for image in no_noise_dataset:
    for algo, pattern in tqdm(ALGO_PATTERN_CONFIG):
        for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
            image = forge(
                image=image,
                demosaicing_algo=algo,
                pattern=pattern,
                jpeg_compression=jpeg_compression,
            )
            votes = get_block_votes(image)

            votes_algo = get_block_votes_on_algo(votes)
            votes_pattern = get_block_votes_on_pattern(votes)
            votes_diag = get_block_votes_on_diag(votes)

            detection_algo, _ = detect_config(votes_algo, num_configs=NUM_ALGO)
            detection_pattern, _ = detect_config(votes_pattern, num_configs=NUM_PATTERN)
            detection_diag, _ = detect_config(votes_diag, num_configs=NUM_DIAG)

            true_algo = ALGO_TO_INDEX[algo]
            true_pattern = PATTERN_TO_INDEX[pattern]
            true_diag = PATTERN_TO_DIAG_INDEX[pattern]

            # Filling algo's values
            if detection_algo == -1: # No detection
                no_detect[jpeg_index][true_algo] += 1/NUM_PATTERN
            elif detection_algo == true_algo: # True detection
                true_detect[jpeg_index][true_algo] += 1/NUM_PATTERN
            else: # False detection
                false_detect[jpeg_index][true_algo] += 1/NUM_PATTERN
                
                if not saving_done:
                    print(f'true_algo = {true_algo}')
                    print(f'true_pattern = {true_pattern}')
                    print(f'detection_algo = {detection_algo}')
                    with open('output/tmp.pkl', 'wb') as f:
                        import pickle
                        print(votes_algo)
                        print(np.unique(votes_algo, return_counts=True))
                        print(detect_config(votes_algo, num_configs=NUM_ALGO))
                        pickle.dump(votes_algo, f)

                    saving_done = True

            # Filling pattern's values
            if detection_pattern == -1: # No detection
                no_detect[jpeg_index][true_pattern + NUM_ALGO] += 1/NUM_ALGO
            elif detection_pattern == true_pattern: # True detection
                true_detect[jpeg_index][true_pattern + NUM_ALGO] += 1/NUM_ALGO
            else: # False detection
                false_detect[jpeg_index][true_pattern + NUM_ALGO] += 1/NUM_ALGO

            # Filling diag's values
            if detection_diag == -1: # No detection
                no_detect[jpeg_index][true_diag + NUM_ALGO + NUM_PATTERN] += 1/NUM_ALGO/2
            elif detection_diag == true_diag: # True detection
                true_detect[jpeg_index][true_diag + NUM_ALGO + NUM_PATTERN] += 1/NUM_ALGO/2
            else: # False detection
                false_detect[jpeg_index][true_diag + NUM_ALGO + NUM_PATTERN] += 1/NUM_ALGO/2

    break

# Plotting
GROUPS = list(DEMOSAICING_ALGOS) + list(PATERNS) + list(DIAGONALS) 
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


plt.xticks(X_AXIS, GROUPS)
plt.xlabel("Groups")
plt.ylabel("Proportions of detections")
plt.title("Detections of the configurations")
plt.legend()
plt.show()
