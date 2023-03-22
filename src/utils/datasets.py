"""
no_noise_images Dataset
dall_e_images Dataset
==============================

References
----------
-   :cite:`colom`: http://mcolom.info/pages/no_noise_images/
-   :cite:`dentan`: https://gist.github.com/DentanJeremie/21bfd925c5234afd15d854135b569bec
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


class Dataset():

    def __init__(
        self,
        name: str,
        url_download: str,
        info_url: str,
        zip_path: Path,
        images_dir: Path,
        unzip_in_images_dir_parent: bool = False,
    ):
        # Attributes
        self.name = name
        self.url_download = url_download
        self.info_url = info_url
        self.zip_path = zip_path
        self.images_dir = images_dir
        self.unzip_in_images_dir_parent = unzip_in_images_dir_parent

        logger.info(f'Initiating a dataset over the {self.name} dataset')
        logger.info(f'More info at {self.info_url}')
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
        if not self.images_dir.exists() or len(list(self.images_dir.iterdir())) <= 2:
            logger.info(f'{self.name} dataset not found')

            if not self.zip_path.exists():
                logger.info(f'{self.name} dataset zip not found, downloading it...')
                response = requests.get(self.url_download, stream=True)
                with self.zip_path.open('wb') as f:
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

            logger.info('Extracting the dataset...')
            try:
                with zipfile.ZipFile(self.zip_path) as zf:
                    if self.unzip_in_images_dir_parent:
                        dst = project.mkdir_if_not_exists(self.images_dir).parent
                    else:
                        dst = project.mkdir_if_not_exists(self.images_dir)
                    zf.extractall(dst)
            except zipfile.BadZipFile:
                logger.info(f'Found corrupted .zip file, deleting it and trying again...')
                self.zip_path.unlink()
                self._get_images()

        else:
            logger.info(f'dataset found at {project.as_relative(self.images_dir)}')

        # Images list
        self._images = sorted([
            item
            for item in self.images_dir.iterdir()
            if '.png' == item.suffix
        ])


no_noise_dataset = Dataset(
    name='no_noise_dataset',
    url_download='https://mcolom.perso.math.cnrs.fr/download/no_noise_images/no_noise_images.zip',
    info_url='https://mcolom.perso.math.cnrs.fr/pages/no_noise_images/',
    zip_path=project.data / 'no_noise_images.zip',
    images_dir=project.data / 'no_noise_images',
)

dall_e_dataset = Dataset(
    name='dall_e_images',
    url_download='https://drive.google.com/uc?export=download&id=1AeVdtoffp0Sgobv-IvgyWmRJsF1oQAX5',
    info_url='https://gist.github.com/DentanJeremie/21bfd925c5234afd15d854135b569bec',
    zip_path=project.data / 'dall_e_images.zip',
    images_dir=project.data / 'dall_e_images',
    unzip_in_images_dir_parent=True,
)