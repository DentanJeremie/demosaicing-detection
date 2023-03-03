import numpy as np
import skimage.measure

from src.detect.forge import forge, get_forgery_config

def get_block_votes(
    image: np.ndarray,
) -> np.ndarray:
    """TO BE COMPLETED
    """
    stacked_residuals = []
    for algo, pattern in get_forgery_config():
        forgery = forge(image, demosaicing_algo=algo, pattern=pattern, inplace=False)
        residual = np.abs(image - forgery)
        residual_one_channel = np.mean(residual, axis=2)
        residual_one_channel_by_block = skimage.measure.block_reduce(
            residual_one_channel,
            (2,2),
            np.mean,
        )
        stacked_residuals.append(residual_one_channel_by_block)

    stacked_residuals = np.array(stacked_residuals)
    return np.argmin(stacked_residuals, axis=0)

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