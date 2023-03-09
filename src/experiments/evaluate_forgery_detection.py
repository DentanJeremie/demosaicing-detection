import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.constants import *
from src.utils.logs import logger
from src.utils.pathtools import project
from src.forgery_tools.forge import generate_forged_image, generate_unforged_image
from src.forgery_tools.detect import detect_forgery
from src.forgery_tools.vote import get_block_votes

NUM_IMAGE_TESTED = 1000
NUM_CONFIG = 2


def evaluate_forgery_detection(
    nfa_threshold_list: t.List[float],
):
    # Init
    num_nfa = len(nfa_threshold_list)

    # Generating the datasets
    forged_datasets = [list() for jpeg_compression in JPEG_COMPRESSION_FACTORS]
    unforged_datasets = [list() for jpeg_compression in JPEG_COMPRESSION_FACTORS]

    logger.info(f'Generating 3x{NUM_IMAGE_TESTED} forged images with various JPEG compression factors')
    for image_count in tqdm(range(NUM_IMAGE_TESTED)):
        for jpeg_count, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
            forged_datasets[jpeg_count].append(
                generate_forged_image(jpeg_compression)
            )

    logger.info(f'Generating 3x{NUM_IMAGE_TESTED} unforged images with various JPEG compression factors')
    for image_count in tqdm(range(NUM_IMAGE_TESTED)):
        for jpeg_count, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
            unforged_datasets[jpeg_count].append(
                generate_unforged_image(jpeg_compression)
            )

    # Initializing the counts of true detections / false detections / no detections
    # true_detect = [array_for_no_jpeg, array_for_jpeg95, array_for_jpeg90]
    # array_for_jpeg95 is the count num_nfa * [true_detect_forgery, true_detect_noforge] for each value of the NFA threshold
    # bar_bottom is just for stacking the bar charts true_detect, no_detect, and false_detect
    true_detect = [
        np.zeros((num_nfa, NUM_CONFIG))
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]
    no_detect = [
        np.zeros((num_nfa, NUM_CONFIG))
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]
    false_detect = [
        np.zeros((num_nfa, NUM_CONFIG))
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]
    bar_bottom = [
        np.zeros((num_nfa, NUM_CONFIG))
        for jpeg in JPEG_COMPRESSION_FACTORS
    ]

    # Filling the variables

    for jpeg_index, sub_forged_dataset in enumerate(forged_datasets):
        logger.info(f'Processing the forged images with JPEG={JPEG_COMPRESSION_FACTORS[jpeg_index]}')
        for image in tqdm(sub_forged_dataset):
            votes = get_block_votes(image)
            is_forged, log_nfa = detect_forgery(
                votes=votes,
                num_configs=len(ALGO_PATTERN_CONFIG),
                windows_size=FORGERY_DETECTION_WINDOWS_SIZE,
                nfa_threshold=None,
            )

            for nfa_index, nfa_threshold in enumerate(nfa_threshold_list):
                if is_forged and 10**(-log_nfa) < nfa_threshold: # True detection forgery
                    true_detect[jpeg_index][nfa_index, 0] += 1/NUM_IMAGE_TESTED
                else:
                    no_detect[jpeg_index][nfa_index, 0] += 1/NUM_IMAGE_TESTED # No detection forgery 

    for jpeg_index, sub_unforged_dataset in enumerate(unforged_datasets):
        logger.info(f'Processing the unforged images with JPEG={JPEG_COMPRESSION_FACTORS[jpeg_index]}')
        for image in tqdm(sub_unforged_dataset):
            votes = get_block_votes(image)
            is_forged, log_nfa = detect_forgery(
                votes=votes,
                num_configs=len(ALGO_PATTERN_CONFIG),
                windows_size=FORGERY_DETECTION_WINDOWS_SIZE,
                nfa_threshold=None,
            )

            for nfa_index, nfa_threshold in enumerate(nfa_threshold_list):
                if is_forged and 10**(-log_nfa) < nfa_threshold: # False detection no forgery
                    false_detect[jpeg_index][nfa_index, 1] += 1/NUM_IMAGE_TESTED
                else:
                    true_detect[jpeg_index][nfa_index, 1] += 1/NUM_IMAGE_TESTED # True detection no forgery 

    # Plotting
    GROUPS = ['Forged', 'Unforged']
    X_AXIS = np.arange(len(GROUPS))
    JPEG_BAR_OFFSET = [-0.20, 0.00, +0.20]

    for nfa_index, nfa_threshold in enumerate(nfa_threshold_list):

        # True detections
        for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
            if jpeg_index == 0:
                kwargs = dict(label='True detection')
            else:
                kwargs = dict()

            plt.bar(
                X_AXIS + JPEG_BAR_OFFSET[jpeg_index],
                true_detect[jpeg_index][nfa_index,:],
                width = 0.15,
                bottom=bar_bottom[jpeg_index][nfa_index,:],
                color='#2CD23E',
                **kwargs
            )
            bar_bottom[jpeg_index][nfa_index,:] += true_detect[jpeg_index][nfa_index,:]

        # No detections
        for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
            if jpeg_index == 0:
                kwargs = dict(label='No detection')
            else:
                kwargs = dict()

            plt.bar(
                X_AXIS + JPEG_BAR_OFFSET[jpeg_index],
                no_detect[jpeg_index][nfa_index,:],
                width = 0.15,
                bottom=bar_bottom[jpeg_index][nfa_index,:],
                color='#4149C3',
                **kwargs
            )
            bar_bottom[jpeg_index][nfa_index,:] += no_detect[jpeg_index][nfa_index,:]

        # False detections
        for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS):
            if jpeg_index == 0:
                kwargs = dict(label='False detection')
            else:
                kwargs = dict()

            plt.bar(
                X_AXIS + JPEG_BAR_OFFSET[jpeg_index],
                false_detect[jpeg_index][nfa_index,:],
                width = 0.15,
                bottom=bar_bottom[jpeg_index][nfa_index,:],
                color='#E53629',
                **kwargs
            )
            bar_bottom[jpeg_index][nfa_index,:] += false_detect[jpeg_index][nfa_index,:]

        # Saving figure
        plt.xticks(X_AXIS, [str(item) for item in GROUPS])
        plt.ylim(0, 1.35)
        plt.xlabel("Groups")
        plt.ylabel("Proportions of detections")
        plt.title(f"Detection with NFA threshold = {nfa_threshold}\n[Left, Middle, Right] = [None, JPEG95, JPEG90]")
        plt.legend()
        plt.savefig(project.output / f'forgery_detection_nfa_{nfa_threshold}.png')
        plt.close()

        # Saving detections counts
        data = [
            [
                f'{config_name}_{jpeg_compression}',
                NUM_IMAGE_TESTED*true_detect[jpeg_index][nfa_index, config_index],
                NUM_IMAGE_TESTED*no_detect[jpeg_index][nfa_index, config_index],
                NUM_IMAGE_TESTED*false_detect[jpeg_index][nfa_index, config_index],
            ]
            for config_index, config_name in enumerate(GROUPS)
            for jpeg_index, jpeg_compression in enumerate(JPEG_COMPRESSION_FACTORS)
        ]
        pd.DataFrame(
            data,
            columns = ['config_jpeg', 'True detect', 'No detect', 'False detect']
        ).to_csv(project.output / f'forgery_detection_nfa_{nfa_threshold}.csv', index=False)

if __name__ == '__main__':
    evaluate_forgery_detection([1, 1e-2, 1e-8])



