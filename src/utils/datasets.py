"""
no_noise_images Dataset
==============================

References
----------
-   :cite:`colom`: http://mcolom.info/pages/no_noise_images/
"""

from pathlib import Path
import sys
import typing as t
import zipfile

import colour
import numpy as np
import requests

from src.utils.pathtools import project
from src.utils.logs import logger

URL_DOWNLOAD = 'https://mcolom.perso.math.cnrs.fr/download/no_noise_images/no_noise_images.zip'
ZIP_PATH = project.data / 'no_noise_images.zip'
IMAGES_DIR = project.data / 'no_noise_images'


class Dataset():

    def __init__(self):
        logger.info(f'Initiating a dataset over the no_noise_images dataset')
        logger.info('More info at http://mcolom.info/pages/no_noise_images/')
        self._images: t.List[Path] = None

        # Iterator
        self._iterator_start = 0
        self._iterator_start_memory = list()
   
    @property
    def images(self) -> t.List[Path]:
        if self._images is None:
            self._get_images()
        return self._images
    
    def __getitem__(self, idx) -> np.ndarray:
        """Returns the idx-th image of the dataset as a RGB array of shape
        HEIGHT, WIDTH, 3, where each pixel is a float32 between 0 and 1.
        """
        return colour.io.read_image(
            self.images[idx]
        )
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __iter__(self):
        self._iterator_start_memory.append(self._iterator_start)
        self._iterator_start = 0
        return self
    
    def __next__(self) -> np.ndarray:
        if self._iterator_start < len(self):
            self._iterator_start += 1
            return self[self._iterator_start - 1]
        
        self._iterator_start = self._iterator_start_memory.pop()
        raise StopIteration

# -------------------- DOWNLOAD ---------------------

    def _get_images(self):
        """Checks that the datasets are correctly downloaded"""
        if not IMAGES_DIR.exists() or len(list(IMAGES_DIR.iterdir())) <= 2:
            logger.info('no_noise_images dataset not found')

            if not ZIP_PATH.exists():
                logger.info('no_noise_images dataset zip not found, downloading it...')
                response = requests.get(URL_DOWNLOAD, stream=True)
                with ZIP_PATH.open('wb') as f:
                    dl = 0
                    total_length = response.headers.get('content-length')
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\rProgression: [%s%s]" % ('=' * done, ' ' * (50-done)) )    
                        sys.stdout.flush()

                sys.stdout.write('\n')

            logger.info('Extracting no_noise_images...')
            try:
                with zipfile.ZipFile(ZIP_PATH) as zf:
                    zf.extractall(project.mkdir_if_not_exists(IMAGES_DIR))
            except zipfile.BadZipFile:
                logger.info(f'Found corrupted .zip file, deleting it and trying again...')
                ZIP_PATH.unlink()
                self._get_images()

        else:
            logger.info(f'no_noise_images found at {project.as_relative(IMAGES_DIR)}')

        # Images list
        self._images = sorted([
            item
            for item in IMAGES_DIR.iterdir()
            if '.png' == item.suffix
        ])

dataset = Dataset()